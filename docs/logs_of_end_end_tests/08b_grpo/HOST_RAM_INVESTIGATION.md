# Host/CPU-RAM investigation — why OpenRLHF RLHF OOMs on this box, and does it really scale to ~1TB?

**Date:** 2026-07-20 (started; empirical section complete, upstream-citation section pending)
**Question being answered:** The role-ladder OOM report concluded that PPO holds "5 separate full
model copies in host RAM (~13.8 GB for a 0.5B model)". Taken literally that implies a 40B model
would need ~1 TB of host RAM, which cannot be right (OpenRLHF is run on 40B+ routinely). So: is the
per-role-copy behaviour correct, and where is the host RAM actually going?

## Empirical measurements on THIS machine (single B70, venv-tf)

Measured with `resource.getrusage(RUSAGE_SELF).ru_maxrss` (peak host RSS) + `torch.xpu.memory_allocated()`.

### 1. Fixed per-process overhead (before any model)
| Stage | Host RSS |
|---|---|
| bare Python | 0.01 GB |
| + import torch (CPU) | 0.52 GB |
| + torch.xpu init | 0.60 GB |
| + ray + deepspeed imported | **0.84 GB** |

=> **~0.84 GB of libraries is paid in EVERY Ray role process**, before a single weight is loaded.
Ray actors are separate OS processes, so this overhead does NOT amortize — it multiplies by role count.

### 2. Model weights in CPU RAM
0.5B (Qwen2.5-0.5B-Instruct), bf16: **~1.0 GB** (matches 0.494B params × 2 bytes = 0.99 GB). Correct/expected.

### 3. THE KEY FINDING — moving the model to the XPU does NOT free the host copy
| Stage | Host RSS | XPU allocated |
|---|---|---|
| weights in CPU | 0.77 GB | 0 |
| **after `.to("xpu")`** | **1.68 GB** | 1.00 GB |
| after one forward pass | 1.87 GB | 1.10 GB |

Host RAM went UP by ~0.9 GB at the same time the XPU allocated 1.0 GB. The Intel Level-Zero/SYCL
runtime appears to keep a **host-side mirror/backing of device memory**; the original CPU copy is
not released. So each role's resident host RAM ≈ 0.84 (libs) + ~1.0 (CPU copy) + ~1.0 (host-side
device backing) [+ optimizer state for actor/critic]. This reconstructs the role-ladder numbers:
ReferenceModelActor 2.34 GB, RewardModelActor 2.01 GB, Actor/Critic 2.9-3.2 GB (extra = optimizer).

## Answering the actual questions

### Q: Is "one model copy per role" real OpenRLHF, or a test artifact?
REAL. actor / critic / reference / reward are genuinely separate Ray actors, each in its own process,
each loading its own model. That is the production `train_ppo_ray.py` design. NOT fabricated.

### Q: Does this mean a 40B model needs ~1 TB of host RAM? — NO. The linear extrapolation is WRONG.
The 0.5B test held the FULL model in every process because we ran ZeRO-0/2 (no parameter sharding).
On real multi-GPU deployments OpenRLHF uses **ZeRO-3**, which shards parameters across the N GPUs:
each process holds only **1/N of the weights**, loaded via `deepspeed.zero.Init()`. Confirmed in code:
`openrlhf/models/actor.py:104-106` — when `zero_stage == 3`, it builds `HfDeepSpeedConfig(ds_config)`
so `from_pretrained` streams+shards at load time (each rank materializes only its shard, never the
whole model in one process). So 40B on 8 GPUs ≈ 5B params resident per process, not 40B. Per-role
cost is DIVIDED by world size on real clusters, not multiplied. [Upstream citation pending from research agent.]

### Q: Why did WE hit it at 0.5B then?
Because we deliberately forced ALL roles onto ONE GPU (`--train.colocate_all`) with ZeRO-0/2 (no
sharding — pointless on 1 GPU) on a 31 GB-RAM box. That is the smallest-hardware corner case, the
opposite of OpenRLHF's normal target (multi-GPU node with hundreds of GB host RAM). The ceiling is
a property of THIS box + this deliberate config, not of the library.

### Q: Is the behaviour "correct"? — Two parts:
1. Per-role model copies: correct and by design (mitigated by ZeRO-3 sharding on real deployments).
2. **XPU host-side device mirror not freeing after `.to(device)`: SUSPICIOUS, likely XPU/Level-Zero
   specific, and is the leading explanation for the ~24 GB unaccounted host RAM in the OOM.** This is
   the part that needs the deferred per-step profiling to confirm fixed-2x-cost vs a per-step leak.

## What is NOT yet proven (deferred, needs a profiling run)
- Whether host RAM stays flat (~2x fixed cost) or GROWS per GRPO step (a leak). Death occurred at the
  NEXT experience-making phase, which is consistent with growth. Plan: run 1.5B GRPO logging `free -h`
  + `xpu-smi` per step; leak => grows monotonically, staging => one-time spike then flat.
- Whether `low_cpu_mem_usage=True` (OpenRLHF only sets it in lora_combiner.py, NOT in the main
  actor/model load path) would materially reduce the load-time transient. Worth testing.

## Upstream references (OpenRLHF README + HF/DeepSpeed docs) — CONFIRMED

Sources: OpenRLHF README (https://github.com/OpenRLHF/OpenRLHF),
`openrlhf/models/actor.py`, HF DeepSpeed docs (https://huggingface.co/docs/transformers/main/en/deepspeed),
HF model-memory anatomy (https://huggingface.co/docs/transformers/en/model_memory_anatomy).

1. **OpenRLHF publishes NO minimum host-RAM number.** Only a Docker `--shm-size="10g"` flag. Its
   examples target **8× A100 80GB** ("For 8x A100 (80GB), try Hybrid Engine"); states role-separation
   across GPUs enables **"70B+ parameters."** So all our host-RAM ceiling numbers are OUR empirical
   findings on a much smaller box, not a documented limit.

2. **Normal multi-GPU path keeps weights in GPU VRAM, sharded — NOT host RAM.** README: Ray
   "separates the Actor, Reward, Reference, and Critic models across different GPUs." Built on
   "DeepSpeed ZeRO-3." Each role holds its model in its own GPU's VRAM.

3. **The "40B => ~1TB host RAM" claim is FALSE as stated.** HF docs: ZeRO-3 shards parameters,
   gradients AND optimizer state across GPUs (each GPU holds only its shard, in VRAM). Full Adam
   mixed-precision training = ~18 bytes/param (6 weights + 8 optimizer + 4 grad). 40B × 18 ≈ ~720 GB
   of TRAINING STATE — but divided across all data-parallel GPUs in **VRAM** (e.g. ~45 GB/GPU on 16
   GPUs), NOT ~1TB of host RAM. Host RAM only balloons to ~model size **if you deliberately enable
   CPU offload** (`--ds.adam_offload` / ZeRO offload_param).

4. **`--ds.adam_offload` is the one knob that DELIBERATELY moves state to host RAM** (~8 bytes/param
   of optimizer state to CPU). On a RAM-constrained box it INCREASES host pressure — README even says
   "Disable `--ds.adam_offload`" when memory is ample. (Matches our finding that removing it got us
   further — GRPO_OOM_FINDINGS.md, attempt #3.)

5. **OpenRLHF does NOT set `low_cpu_mem_usage`** in the main actor/model load path (only in
   lora_combiner.py). It relies on `HfDeepSpeedConfig(ds_config)` built BEFORE from_pretrained when
   zero_stage==3, which triggers ZeRO-3 `zero.Init` to shard params AT LOAD TIME so the full model is
   never materialized in one process. **BUT this only kicks in at ZeRO-3.** At ZeRO-0/2 (what we ran
   on 1 GPU), there is NO load-time sharding => full model resident per process. This matches our
   measurements.
   - REFINEMENT (transformers v4.20+): even without `low_cpu_mem_usage`, from_pretrained uses a
     **meta-device skeleton** by default, so peak load-time host RAM is ~**1×** model size, not the
     old ~2× double-copy. So at ZeRO-0/2 each process holds ~1× the model in host RAM (not 2×) —
     consistent with our measured ~1 GB CPU copy for the 0.5B model. Sources:
     https://huggingface.co/docs/transformers/main/en/big_models ,
     https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference
   - So the resident host cost per role ≈ libs (~0.84) + ~1× weights (CPU copy) + XPU host-side
     device backing (~1×, the suspicious part) [+ optimizer for actor/critic]. The double-copy is NOT
     the culprit; the XPU host-side backing is the thing to investigate.

6. **Documented OOM levers (README):** toggle `--ds.adam_offload`; use/disable `--train.colocate_*`
   options ("Disable all --colocate_* options" on OOM — i.e. spread roles back across more GPUs).

### Reconciling with OUR result
We ran the configuration that is the WORST case for host RAM and the OPPOSITE of OpenRLHF's design
target: single GPU, ZeRO-0/2 (no param sharding — correct for 1 GPU but means full model per process),
all roles colocated on one box, 31 GB RAM. On the intended 8×80GB-class node with ZeRO-3, weights are
sharded into VRAM and this host-RAM ceiling is essentially never reached. Our OOM is a small-hardware
corner case, correctly explained — NOT an OpenRLHF defect and NOT evidence that large models need TB-scale RAM.

The ONE remaining XPU-specific open item (host copy not freed after `.to(xpu)`; possible per-step
growth) is unaffected by the above and still needs the deferred per-step profiling run.

## MAJOR UPDATE — this is a KNOWN upstream OpenRLHF host-RAM issue, likely NOT XPU-specific

GitHub-issue search found this exact symptom reported on NVIDIA hardware by multiple users. Our
"died at the NEXT experience-making step" pattern is the signature of a per-step host-RAM leak that
OpenRLHF users hit on CUDA too. Key issues:

- **#1065 "Why CPU Memory Is Used More and More?"** (OPEN, unresolved) — verbatim: "the rss of RAM
  occupied indeed increases w.r.t. steps"; "At each saving point, the cache becomes obviously larger."
  User OOM'd after >1k steps on NVIDIA. Disabling `--adam_offload` slowed growth; checkpoint-save
  growth remained. Maintainer suggested `gc.collect()` + `ray.internal.free_objects()` per step, and
  filed DeepSpeed issue #7370. **This is our symptom, on CUDA — so it is NOT primarily an XPU bug.**
  https://github.com/OpenRLHF/OpenRLHF/issues/1065
- **#1013 / #970** — `--vllm_enable_sleep` and older vLLM versions repeatedly implicated in host-RAM
  growth; #1013's fix was **removing `--vllm_enable_sleep`**. **NOTE: our runs USE `--vllm.enable_sleep`**
  — so vLLM sleep is a prime suspect for our growth too. (Trade-off: we added sleep to fix a *VRAM*
  init failure; it may be costing host RAM.) https://github.com/OpenRLHF/OpenRLHF/issues/1013 , /970
- **#911** (CLOSED, confirmed fix) — 7B REINFORCE++ on 8×A800/250GB RAM OOM'd at 239/251GB. Fix that
  worked: **"zero3 + adam offload, OR Hybrid engine + deepspeed sleep."** https://github.com/OpenRLHF/OpenRLHF/issues/911
- **#171** — loading all actor/critic/reward/ref for a 13B model needed **450GB host RAM** — confirms
  the multi-model host-RAM footprint is real and large even on the intended hardware.
- **#1139** — custom `reward_func` host-RAM leak; fixes: lower `OPENRLHF_ASYNC_NUM_TASKS` (default 128
  → 4-8), and fix un-freed variables in the reward function. https://github.com/OpenRLHF/OpenRLHF/issues/1139
- **#936** — `RAY_DEBUG_DISABLE_MEMORY_MONITOR=1` only silences Ray's killer; the kernel OOM killer
  still fires — NOT a real fix (we should not rely on raising the Ray threshold either).

### Revised conclusion
The host-RAM exhaustion is **substantially a known OpenRLHF/vLLM-sleep/checkpoint-save behaviour seen
on CUDA too**, amplified on our box by (a) tiny 31 GB RAM, (b) single-GPU no-sharding config, and
possibly (c) an additional XPU host-side device-backing cost. It is NOT uniquely an XPU defect, and
NOT evidence that large models need ~1TB RAM (that's the ZeRO-3-shards-to-VRAM story, sections above).

### Concrete levers to try when we resume (in rough priority order)
1. **Drop `--vllm.enable_sleep`** and instead give vLLM a fixed low `--vllm.gpu_memory_utilization`
   (the #1013 fix). This is the highest-signal experiment given we currently use sleep.
2. Per-step `gc.collect()` + `ray.internal.free_objects()` (#1065 maintainer suggestion).
3. Lower `OPENRLHF_ASYNC_NUM_TASKS` (128 → 4-8) (#1139).
4. Confirm `--ds.adam_offload` is OFF (it is, in our config) — offload worsens host RAM here.
5. THEN the per-step `free -h`/`xpu-smi` profiling to see if any XPU-specific residual remains after
   the above upstream mitigations are applied.
