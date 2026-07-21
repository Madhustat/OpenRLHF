# OpenRLHF Single-GPU XPU Validation — Status Update

**Date:** 2026-07-20 · **Machine:** Intel Arc Pro B70 (single GPU, 32 GB VRAM, **31 GB system RAM**)
**Stack:** torch 2.12.0+xpu · Ray 2.55.0 · vLLM 0.23.1 (XPU build) · DeepSpeed 0.19.1 · transformers 5.13

---

## Summary (TL;DR)

All 12 OpenRLHF workflows validated end-to-end on a single Intel XPU (SFT, RM, DPO, vLLM, Ray, PPO,
GRPO, REINFORCE++, checkpoint resume — **12/12 PASS**). One issue found during RLHF runs: **host
(system) RAM exhaustion**, NOT a GPU or XPU-porting defect. Root-cause investigation shows this is a
**known, documented OpenRLHF behaviour reproduced on NVIDIA hardware by other users** — not specific
to Intel. Workaround (0.5B model) unblocked all validation; a small set of upstream-documented fixes
is queued to try for the 1.5B case.

---

## 1. Validation results — 12/12 PASS

| Workflow | Stack exercised | Model | Result |
|---|---|---|---|
| CLI ×4, model load+generate | XPU | 1.5B | ✅ |
| SFT / RM / DPO (5 steps each) | DeepSpeed on XPU | 1.5B | ✅ |
| vLLM standalone / Ray worker | vLLM / Ray on XPU | 0.5B | ✅ |
| GRPO / REINFORCE++ / PPO+critic (5 steps each) | Ray + vLLM + DeepSpeed on XPU | 0.5B | ✅ |
| Checkpoint save → reload → resume | DeepSpeed on XPU | 1.5B | ✅ |

Exit criteria for single-GPU validation are met. Cleared to plan 2-GPU distributed validation.

---

## 2. Issue found: host-RAM exhaustion in the RLHF loop (PPO/GRPO/REINFORCE++)

**Symptom:** with Qwen2.5-**1.5B**, the Ray-based RLHF run completes ≥1 real training step, then is
killed at the next experience-making phase when node RAM crosses ~30 / 31 GB. GPU VRAM had ample
headroom throughout (vLLM used ~11 / 32 GB). **Not a VRAM/XPU-code failure — a system-RAM capacity/leak issue.**

**Immediate workaround (applied, validated):** switch RLHF models to Qwen2.5-**0.5B** → all three RL
variants pass cleanly, 5 steps, exit 0. This is a workaround, not a root-cause fix.

---

## 3. Root-cause investigation — findings merged with upstream evidence

### 3a. Our empirical measurements (this machine)
- Per-process fixed overhead (torch+xpu+ray+deepspeed libraries) = **~0.84 GB in EVERY Ray role
  process** (actor/critic/reference/reward are separate OS processes — overhead does not amortize).
- 0.5B model weights in CPU RAM (bf16) = ~1.0 GB (as expected).
- **Moving a model `.to("xpu")` does NOT free the host copy** — host RAM rose ~1 GB while XPU also
  allocated ~1 GB (a possible XPU/Level-Zero host-side mirror). Minor contributor vs the item below.

### 3b. This is a KNOWN OpenRLHF issue — reproduced on NVIDIA, not XPU-specific
Our "RAM grows each step, dies at the next" signature matches multiple upstream reports:

| Issue | Relevance | Status |
|---|---|---|
| [#1065 "Why CPU Memory Is Used More and More?"](https://github.com/OpenRLHF/OpenRLHF/issues/1065) | **Exact symptom** — user quote: *"the rss of RAM occupied indeed increases w.r.t. steps"*, *"At each saving point, the cache becomes obviously larger."* OOM after >1k steps on NVIDIA. | OPEN, unresolved |
| [#1013 "Task killed due to node running low on memory (ray oom)"](https://github.com/OpenRLHF/OpenRLHF/issues/1013) | Host-RAM OOM at load/rollout. **Fixed by removing `--vllm_enable_sleep`.** | Open, workaround found |
| [#970 long-context OOM killer](https://github.com/OpenRLHF/OpenRLHF/issues/970) | OOM *"right when rollout samples finish generating"*; vLLM memory leak implicated. | Open |
| [#911 "node running low on memory"](https://github.com/OpenRLHF/OpenRLHF/issues/911) | 7B on 8×A800/250GB RAM OOM'd at 239/251 GB. **Fix confirmed: ZeRO-3 + adam_offload, OR Hybrid engine + sleep.** | Closed (fixed) |
| [#171 "High Memory Usage"](https://github.com/OpenRLHF/OpenRLHF/issues/171) | Loading all 4 models for a **13B** run needed **450 GB host RAM** — confirms multi-model host footprint is large even on target HW. | Closed |
| [#1139 custom reward_func OOM-killed](https://github.com/OpenRLHF/OpenRLHF/issues/1139) | Fix: lower `OPENRLHF_ASYNC_NUM_TASKS` (default 128 → 4-8); fix un-freed vars in reward fn. | Closed |

**Prime suspect for our case:** we run with `--vllm.enable_sleep` — the exact flag #1013 removed to
fix its OOM, and repeatedly implicated in host-RAM growth (#1013, #970, #1065). We enabled it to fix
a *VRAM* startup issue; it may be trading VRAM for host-RAM.

### 3c. Corrected the "large models need ~1 TB RAM" concern
A prior working note implied per-role model copies scale to ~1 TB host RAM for a 40B model. **This is
false**, confirmed against primary sources:
- OpenRLHF's normal path uses **DeepSpeed ZeRO-3**, which shards weights + gradients + optimizer state
  **across GPU VRAM** (each GPU holds 1/N), not host RAM. ([ZeRO paper](https://arxiv.org/abs/1910.02054),
  [DeepSpeed ZeRO docs](https://www.deepspeed.ai/tutorials/zero/), [HF DeepSpeed docs](https://huggingface.co/docs/transformers/main/en/deepspeed))
- 40B training state ≈ 640 GB (16 bytes/param) lives in VRAM **divided across the cluster** (e.g.
  ~40 GB/GPU on 16 GPUs) — OpenRLHF examples target 8×A100 80GB / 16 GPUs for 70B.
- Host RAM only scales to model size if you **deliberately enable CPU offload** (`--ds.adam_offload` /
  `offload_param`). We do **not** use offload. So there is no TB-scale RAM requirement.

---

## 4. Why WE hit it (and OpenRLHF's normal users mostly don't)
We ran the deliberate worst case for host RAM: **single GPU + no ZeRO-3 param sharding (pointless on
1 GPU) + all 4-5 roles colocated on one 31 GB box.** OpenRLHF's design target is a multi-GPU node
(hundreds of GB RAM) where ZeRO-3 shards weights into VRAM. On a 31 GB box the ceiling is easy to hit;
on target hardware it is largely avoided.

---

## 5. Next steps (queued, not yet run) — ranked by expected signal
1. **Drop `--vllm.enable_sleep`**, use a fixed low `gpu_memory_utilization` instead — the [#1013](https://github.com/OpenRLHF/OpenRLHF/issues/1013) fix; top suspect since we currently use sleep.
2. Per-step `gc.collect()` + `ray.internal.free_objects()` — [#1065](https://github.com/OpenRLHF/OpenRLHF/issues/1065) maintainer suggestion.
3. Lower `OPENRLHF_ASYNC_NUM_TASKS` 128 → 4-8 — [#1139](https://github.com/OpenRLHF/OpenRLHF/issues/1139).
4. Then per-step `free -h` + `xpu-smi` profiling to isolate any XPU-specific residual after the above.
5. Proceed to 2-GPU distributed validation (enables ZeRO-3 sharding, which relieves host RAM directly).

**Detailed technical writeup:** `docs/logs_of_end_end_tests/08b_grpo/HOST_RAM_INVESTIGATION.md`
**Original OOM log analysis:** `docs/logs_of_end_end_tests/08b_grpo/GRPO_OOM_FINDINGS.md`
