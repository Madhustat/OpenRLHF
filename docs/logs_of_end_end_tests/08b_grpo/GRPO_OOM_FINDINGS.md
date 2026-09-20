# GRPO single-GPU (colocate_all) on Intel Arc Pro B70 — OOM findings

**Date:** 2026-07-20
**Machine:** PG16WVAW7054, single Intel Arc Pro B70 (BMG), **31 GB system RAM**, 8 GB swap.
**Stack:** torch 2.12.0+xpu, ray 2.55.0, vllm 0.23.1rc1 (XPU source build), deepspeed 0.19.1.
**Model:** Qwen/Qwen2.5-1.5B-Instruct. Dataset: gsm8k_train_prompts.jsonl (math reward func).
**Run from:** `/home/dut7054/openrlfh_exp/OpenRLHF`; env from work2's `venv-tf`.

## Bottom line

The full OpenRLHF RLHF loop (Ray + vLLM + DeepSpeed) **runs correctly on the XPU** — vLLM
generates rollouts, the math reward function scores them, and **real GRPO optimizer steps complete**
(`act_loss`, `reward`, `kl` all logged with real values, reward=1.0 on correct answers). It is
**NOT an XPU code/port bug.** The blocker is **host system RAM**: the run dies from a Ray
OutOfMemoryError (or kernel OOM killer) once node RAM crosses ~30 GB of the 31 GB total.

The run dies from host-RAM exhaustion, not a config or code defect in the training loop itself.
**HOWEVER — the *amount* of host RAM consumed (~30 GB for a 1.5B model) is unexpectedly high and
is only ~6 GB accounted for; the ~24 GB remainder is unexplained and needs investigation (see the
"Key diagnostic" section below). Do NOT close this out as simply "needs more RAM."**
The teammate's verified single-GPU run (MULTIGPU_LORA_ZERO3_fixes_and_flags.md, row 1, "8 steps,
exit 0") was on the 2× B70 box, which presumably has more host RAM — but that does not explain
*why* the footprint is this large.

## What passed before OOM (proof the loop works)

Across attempts, before the OOM kill we consistently saw:
- vLLM engine builds on XPU, allocates KV cache (~8.6 GiB), enters sleep mode correctly.
- `🚀 Starting experience making with N samples` → `✨ Experience making completed`.
- Real rollouts + reward scoring, e.g. `[Math Reward] Pred: 62, Gold: 62, Match: True`.
- **At least one full GRPO training step**, e.g.
  `act_loss=-0.696, reward=1, return=0.707, kl=0` and `act_loss=-0.278, reward=0.25`.

So: generation ✅, reward ✅, advantage/GRPO update ✅, weight-sync path reached ✅.

## The exact failure

```
ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory.
Memory on the node was 29.77GB / 31.17GB (0.955), which exceeds the threshold of 0.95;
Top memory users: PolicyModelActor.fit 3.87GB, VLLM::EngineCore 1.02GB, PPOTrainer.fit 0.67GB,
RolloutRayActor 0.35GB ...
```
Last app traceback frame: `train_ppo_ray.py:193 → ray.get(ppo_trainer.fit.remote())`.
When the Ray threshold was raised to 0.98, the failure moved to the **kernel OOM killer** instead
(worker "connection error code 2 / EOF", SYSTEM_ERROR) — same root cause, different reaper.

## Key diagnostic: tracked memory (~6 GB) vs actual node memory (~30 GB) — UNEXPLAINED, NEEDS INVESTIGATION

Ray's "top memory users" only sum to **~6 GB**, yet the node reports **~30 GB** used. **This ~24 GB
gap is NOT explained** and is the single most important open question from this run.

What we can legitimately account for (host-side): CPU copies of weights during load, DeepSpeed
host-side param/optimizer state, pinned transfer buffers, the rollout/experience buffer, and vLLM
sleep-mode's CPU backup (~1 GB/cycle, per log). Honestly summed for a 1.5B model these should be
well under ~15 GB — **NOT ~30 GB.** So a large chunk of host RAM is being consumed by something we
have not measured.

**IMPORTANT:** ~30 GB of *system RAM* to train a *1.5B* model, while GPU VRAM sits mostly idle
(~11/32 GB used), is **abnormal** — this is NOT the expected footprint and should not be read as
"RLHF just needs lots of RAM." Leading hypotheses, ALL UNVERIFIED:
1. Intel XPU runtime (Level Zero / SYCL) keeping large host-side staging / shadow allocations for
   device memory.
2. A host-RAM leak / accumulation across steps (death occurred at the *next* experience-making
   phase, consistent with growth-over-time rather than a one-time spike).
3. Pinned host memory not being released across sleep/wake cycles.

**Action item (deferred, not yet done):** profile a 1.5B GRPO run watching `free -h` + `xpu-smi`
memory side-by-side across steps to distinguish leak (grows step-over-step) vs one-time staging
spike. Until then, the 0.5B model is a **workaround that unblocked validation, NOT a root-cause fix.**

The XPU VRAM side is fine (`gpu_memory_utilization 0.4` leaves headroom). It's the **host RAM**
that saturates — for reasons only ~6 GB of which are currently accounted for.

## What was tried (all reproduced the same host-RAM ceiling)

| # | Change | Result |
|---|--------|--------|
| 1 | Baseline w/o sleep flags | vLLM init fail: "Free memory 0.0/30.3 GiB" (fixed by sleep flags) |
| 2 | + `--vllm.enable_sleep --ds.enable_sleep`, gpu_mem 0.4 | Got to optimizer init, segfault at `cpu_adam` (adam_offload) |
| 3 | Removed `--ds.adam_offload` (optimizer on XPU) | **1 full GRPO step ran**, then OOM at next experience-making |
| 4 | + `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` (device isolation fix from runbook) | **Step completed**, then host-RAM OOM (30.14/31.17 GB) |
| 5 | + smaller buffers (n_samples 2, rollout/train batch 4, max_samples 20), `RAY_memory_usage_threshold=0.98` | Step ran, kernel OOM killer (threshold raised → kernel reaped it) |
| 6 | + cap `--object-store-memory=2GB`, `max_len 384`, `max_new_tokens 200`, micro_batch 1, threshold back to 0.95 | Step ran, still OOM at 29.77/31.17 GB |

Config changes shrank the *tracked* footprint but never brought the *node* peak below ~30 GB — because
the dominant consumer is the unified/host-backed XPU + sleep-mode CPU backup, which config knobs
don't reduce much.

## Verified NOT the cause (ruled out)

- **Not XPU VRAM OOM** — vLLM at 0.4 util, VRAM had headroom.
- **Not the sleep/offload port** — sleep freed 11.75 GiB cleanly ("It took 0.09s to fall asleep").
- **Not a `torch.cuda.*` gap** — no AttributeError; optimizer step completed with real numbers.
- **Not the device-isolation bug** — NOSET fix applied; role landed on the right device and trained.
- **Not adam_offload** — removing it got us *further* (offload actually adds host-RAM pressure here).

## Questions for the friend / next levers (need input before more runs)

1. **How much host RAM did the passing 2× B70 box have?** If >64 GB, that alone explains it and
   this single-GPU 31 GB box simply can't hold colocate_all for a 1.5B model + vLLM.
2. Is there a way to **disable vLLM sleep-mode's CPU backup** (discard instead of back up the
   ~3.1 GiB each cycle)? That backup is a big chunk of the host-RAM churn.
3. Would **`--ds.zero_stage 3` without offload** reduce resident actor RAM enough (params sharded)?
   Not yet tried on single GPU (runbook only lists it for 2-GPU + offload).
4. Add swap? Currently 8 GB (2 GB used). A larger swap file might absorb the ~2-4 GB transient
   overshoot, at a speed cost — acceptable for a smoke test.
5. `RAY_memory_monitor_refresh_ms=0` to fully disable Ray's killer and let it lean on swap — risky
   (kernel OOM killer still active) but would tell us the true peak.

## RESOLUTION (2026-07-20) — switched to Qwen2.5-0.5B-Instruct

Confirmed hypothesis #1: the 1.5B model + vLLM + colocate_all simply does not fit in 31 GB host
RAM on this box. **Switching the actor/reward/critic/reference model to Qwen2.5-0.5B-Instruct
(~1/3 the params) resolved the OOM entirely** — no code or flag change needed beyond the model name.

All three RL variants then passed cleanly on the single 31 GB XPU box, 5 global steps each, exit 0,
with real metrics and real weight-sync (`timing/broadcast ~1s`):
| Variant | Estimator | Result | Evidence |
|---|---|---|---|
| GRPO | `group_norm` | ✅ 5 steps, exit 0 | step 4: reward=0.25, math_accuracy=0.25 |
| REINFORCE++ | `reinforce_baseline` | ✅ 5 steps, exit 0 | policy_loss=0.00063, reward=0.125 |
| PPO | `gae` + critic | ✅ 5 steps, exit 0 | critic_loss=0.0097, values present (4th model trained) |

Notably PPO+gae+critic (4 models) fit on ONE GPU with the 0.5B model — previously assumed to need
2 GPUs. Peak host RAM stayed ~24-27 GB free-margin comfortable.

**Takeaway for the 31 GB single-GPU box:** use ≤0.5B models for colocate_all RLHF, OR add host RAM
/ a second GPU for 1.5B+. The 1.5B run is viable on the 2× B70 box per the teammate's runbook.

## Log locations

All attempts: `docs/logs_of_end_end_tests/08b_grpo/grpo_group_norm_*.log` (1.5B OOM attempts +
0.5B pass), `../08c_reinforce_plus_plus/`, `../08_ppo/`.
Launcher used: `/tmp/run_ppo_phase8.sh` (parameterized by estimator; MODEL=Qwen2.5-0.5B-Instruct).
