# Multi-GPU device isolation, LoRA+vLLM, ZeRO-3, and single-GPU colocation — fixes & required flags

**Verified 2026-07-20 on the 2× Intel Arc Pro B70 (BMG) box.** Stack: `torch 2.12.0+xpu`,
`ray 2.55.0` (venv), `vllm 0.23.1rc1` (source build at `openrlhf_exp/vllm-src`), `peft 0.19.1`.

This doc covers four things that were blocking real multi-config PPO on XPU, and exactly how to
run each. **Two are source-code fixes** (also added to `PORTING_openrlhf_for_xpu.md` as items 15
& 16); **two are "use the right flags", not code bugs.** Read this first when a 2-GPU or LoRA or
single-GPU PPO run fails.

---

## TL;DR — what works and how to run it

| Config | Works? | What was needed |
|---|---|---|
| **2-GPU, no colocation** (actor on GPU A, vLLM on GPU B) | ✅ | Code fix #16 (`ZE_AFFINITY_MASK`) + launch env var (below) |
| **2-GPU LoRA** | ✅ | Code fix #15 (LoRA-merge-before-broadcast) |
| **2-GPU ZeRO-3** | ✅ | Add `--ds.adam_offload` (not a bug — on-device optimizer runs out of resources) |
| **Single-GPU colocate_all** | ✅ | Add `--vllm.enable_sleep --ds.enable_sleep` (not a bug — colocation is designed to need sleep mode) |

All four verified with full 8-step Qwen2.5-1.5B GSM8K runs (real reward/accuracy, clean exit).

---

## Fix #16 (code) — actor + vLLM engine both land on `xpu:0` in a 2-GPU Ray setup

**Symptom:** with `--actor.num_gpus_per_node 1` + a separate vLLM engine (no `--train.colocate_all`),
both the DeepSpeed actor and the vLLM engine land on physical `xpu:0`; GPU 1 stays idle; you get
OOM on weight broadcast, or vLLM init fails citing `xpu:0`.

**Root cause (two layers):**
1. On XPU the device-isolation env var (`ONEAPI_DEVICE_SELECTOR` / `ZE_AFFINITY_MASK`) is only
   honored by the Level-Zero runtime if set **before** the runtime initializes. Ray sets
   `ONEAPI_DEVICE_SELECTOR` *inside* the already-running worker (`ray/_private/utils.py`
   `set_visible_accelerator_ids`), which on XPU is too late → worker still sees both devices and
   both roles default to logical index 0.
2. OpenRLHF's vLLM path only ever set `CUDA_VISIBLE_DEVICES`, which the XPU runtime ignores.
   vLLM's XPU platform reads **`ZE_AFFINITY_MASK`** (`vllm/platforms/xpu.py`:
   `device_control_env_var = "ZE_AFFINITY_MASK"`), not `CUDA_VISIBLE_DEVICES`.

**Fix (code):** in `openrlhf/trainer/ray/vllm_engine.py`, `_configure_device_env`, the NOSET
branch now also sets `ZE_AFFINITY_MASK` to the Ray-assigned GPU id, before `import vllm` /
first XPU call:
```python
# BEFORE:
elif ray_noset_visible_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
# AFTER:
elif ray_noset_visible_devices():
    gpu_id = str(ray.get_gpu_ids()[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["ZE_AFFINITY_MASK"] = gpu_id   # XPU reads this, not CUDA_VISIBLE_DEVICES
```
The actor side already binds correctly under NOSET (`launcher.py` sets `LOCAL_RANK =
ray.get_gpu_ids()[0]` → `deepspeed.py` `set_device_index`), so no actor code change was needed.

**Required launch env var** (multi-GPU, no colocation): export before `ray start` so all workers
inherit it, which makes both `ray_noset_visible_devices()` branches take the correct path:
```bash
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
```
Why NOSET: it tells Ray *not* to attempt its (too-late, ineffective on XPU) device masking, and
OpenRLHF then pins each role to its Ray-assigned GPU id explicitly — which works because devices
stay unmasked and `set_device(gpu_id)` selects the right physical XPU.

**Distinguishing the two physical XPUs** (names are identical, use UUID): gpu 0 →
`868023e2-…-1800-…`, gpu 1 → `868023e2-…-5400-…`.

**Do NOT install `dpctl` to "help" Ray detect XPUs** — it pulls Intel runtime libs at 2026.x
that conflict with torch 2.12.0+xpu (pins 2025.3.2) and breaks torch. The NOSET + ZE_AFFINITY_MASK
approach needs no dpctl.

---

## Fix #15 (code) — LoRA weight-sync to vLLM: `ValueError: no module or parameter named 'base_model'`

**Symptom:** `--ds.lora.rank > 0` with `train_ppo_ray`; a full epoch trains, then the weight
broadcast to vLLM fails with `ValueError: There is no module or parameter named 'base_model' in
Qwen2ForCausalLM`. (Matches OpenRLHF GitHub **issue #991**, which was open/unsolved there.)

**Root cause:** OpenRLHF lost its LoRA-merge-before-broadcast logic in a refactor. `broadcast_to_vllm`
iterated `model.named_parameters()`, but under LoRA `model` is a `PeftModel`, so names come out as
`base_model.model.model.layers…` and the LoRA adapter deltas were never folded into the base
weights. vLLM only knows `model.layers…` and expects merged weights.

**Fix (code):** restored the merge approach from OpenRLHF's own closed **PR #869** ("support
lora+vllm"), adapted to the current (post-refactor, dotted-config, XPU-gloo) code, in
`openrlhf/trainer/ray/ppo_actor.py` `broadcast_to_vllm`:
- Added helper `_get_leaf_modules(root, use_lora)` — returns leaf modules (or LoRA-wrapped
  layers), wrapping isolated params, and skipping `lora_`/`base_layer` bookkeeping params.
- In `broadcast_to_vllm`: if `isinstance(model, PeftModel)`, unwrap `model.base_model.model`
  (strips the `base_model.` name prefix); for each LoRA leaf module: `GatheredParameters` (if
  ZeRO-3) → `module.merge(safe_merge=True)` → `lora_model._replace_module(...)` onto a fake
  parent to get the merged base layer → broadcast the merged weights with the correct name →
  `module.unmerge()` on exit (via `ExitStack`). Non-LoRA path is the unchanged original logic in
  the `else` branch.

**Gating / safety:** the LoRA branch only runs for a `PeftModel` (i.e. only when `lora.rank > 0`);
full fine-tuning takes the unchanged `else` branch. Verified by a full-FT 8-step run passing after
the change. `peft 0.19.1` provides all APIs used (`_replace_module`, `merge`, `unmerge`,
`get_base_layer`).

**Run it:** add `--ds.lora.rank 32 --ds.lora.alpha 32` to any working PPO command.

---

## ZeRO-3 — not a bug, needs `--ds.adam_offload`

**Symptom:** `--ds.zero_stage 3` (without offload) — vLLM/rollout/reward all init fine, then the
actor optimizer step fails: `deepspeed/ops/adam/fused_adam.py → RuntimeError: level_zero backend
failed with error 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)`.

**Cause:** the on-device fused Adam optimizer exhausts XPU resources. **Not an isolation or code
bug.** Adding `--ds.adam_offload` (moves optimizer state to CPU) fixes it.

**Run it:** `--ds.zero_stage 3 --ds.adam_offload`. Verified 8 steps, clean.

---

## Single-GPU / any colocation — not a bug, needs sleep-mode flags

**Symptom:** `--train.colocate_all` (the only option on a real single-GPU box) fails at vLLM init:
`AssertionError: Error in memory profiling. Initial free memory 20.58 GiB, current free memory
22.54 GiB. … other processes sharing the same container release GPU memory while vLLM is
profiling.` (`vllm/v1/worker/gpu_worker.py`). Note free memory *rose* — this is **not** an OOM.

**Cause:** under colocation the actor/reference/vLLM share one GPU. vLLM's memory profiler asserts
that free memory does not *increase* mid-profile, but a concurrently-initializing colocated process
(optimizer/actor) releases memory on the shared device, tripping the assert. OpenRLHF's colocation
is **designed** to use vLLM sleep mode so the engine releases the GPU when the actor needs it — the
failure only happens when the sleep flags are omitted. **Not a code bug.**

**Run it (single-GPU or any colocate_all):** add both flags:
```bash
--vllm.enable_sleep --ds.enable_sleep
```
Plus start Ray with 1 GPU and lower vLLM headroom since 3 roles share the device:
```bash
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
venv-tf/bin/ray start --head --num-gpus=1
# … --train.colocate_all --train.colocate_actor_ref --vllm.enable_sleep --ds.enable_sleep \
#    --vllm.gpu_memory_utilization 0.4 …
```
Verified: true single-GPU (`--num-gpus=1`) colocate_all + sleep flags → 8 steps, exit 0, no
memory-profiling assert, accuracy up to 0.90.

---

## Verified capability matrix (all full 8-step Qwen2.5-1.5B GSM8K, 2026-07-20)

| # | Config | Flags (beyond the common set) | Result |
|---|---|---|---|
| 1 | Single-GPU colocate_all | `--num-gpus=1` + `--vllm.enable_sleep --ds.enable_sleep --vllm.gpu_memory_utilization 0.4` | ✅ 8 steps |
| 2 | 2-GPU full FT (no-colo) | `--ds.zero_stage 2 --ds.adam_offload --vllm.gpu_memory_utilization 0.8` | ✅ 8 steps |
| 3 | 2-GPU LoRA | as #2 + `--ds.lora.rank 32 --ds.lora.alpha 32` | ✅ 8 steps |
| 4 | 2-GPU ZeRO-3 | `--ds.zero_stage 3 --ds.adam_offload` | ✅ 8 steps |

Common set (all runs): `--vllm.num_engines 1 --vllm.tensor_parallel_size 1 --vllm.sync_backend gloo
--vllm.enforce_eager --algo.advantage.estimator group_norm --algo.kl.init_coef 0
--rollout.n_samples_per_prompt 4 --rollout.batch_size 8 --reward.remote_url
examples/python/math_reward_func.py --ds.attn_implementation sdpa`, plus multi-GPU runs export
`RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` before `ray start`.

## Two operational gotchas that cost debugging time

1. **Ray version shadow:** `which ray` may resolve to `/home/sdp/miniforge3/bin/ray` (a *different*
   version) which shadows the venv's ray → cluster "Version mismatch" RuntimeError at actor
   startup. Always start Ray with `venv-tf/bin/ray start …` and launch training with
   `venv-tf/bin/python`.
2. **`tensor_parallel_size > 1` is untested** on XPU here — it would need `ZE_AFFINITY_MASK` set to
   the full comma-list of assigned GPU ids per worker (fix #16 currently sets a single id, correct
   for TP=1).

## Files changed (both also listed in `PORTING_openrlhf_for_xpu.md` items 15–16)
- `openrlhf/trainer/ray/vllm_engine.py` — `_configure_device_env`: set `ZE_AFFINITY_MASK`.
- `openrlhf/trainer/ray/ppo_actor.py` — `broadcast_to_vllm`: LoRA merge-before-broadcast + new
  `_get_leaf_modules` helper.
