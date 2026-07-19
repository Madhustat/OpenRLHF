# OpenRLHF Phase-1 XPU Enablement — Ray & PPO Findings

Environment: `~/madhu/work2_with_transformers_native/OpenRLHF/venv-tf`
Hardware: Intel Arc Pro B70 (Battlemage), Ubuntu 24.04.3
Validated: 2026-07-07

This document covers Ray functional validation and the PPO (`train_ppo_ray.py`) investigation
— the two components in the "not yet exercised" table from Phase-1's earlier status
(`DeepSpeed: confirmed working`, `Ray: not exercised`, `vLLM: not exercised`).

---

## Part 1 — Ray: confirmed working on Intel XPU, with one real caveat

### Basic functional test
```python
import ray
ray.init(num_cpus=2, ignore_reinit_error=True)

@ray.remote
def add(a, b):
    return a + b

result = ray.get(add.remote(3, 4))  # -> 7, SUCCESS
```
Confirmed: Ray core (cluster init, remote task dispatch, `ray.get()`) works correctly.

### XPU visibility from inside a Ray worker
```python
@ray.remote
def check_xpu():
    import torch
    return {'xpu_available': torch.xpu.is_available(), 'device_count': torch.xpu.device_count(), ...}

ray.get(check_xpu.remote())
# -> {'xpu_available': True, 'device_count': 1, 'device_name': 'Intel(R) Arc(TM) Pro B70 Graphics'}
```
Confirmed: a Ray worker (separate subprocess Ray spawns) correctly sees the Intel GPU — not
just the parent process.

### GPU-resource scheduling — works, but only with an important caveat
```python
ray.init(num_cpus=2, num_gpus=1, ignore_reinit_error=True)

@ray.remote(num_gpus=1)
def gpu_task():
    ...
```
This worked and correctly scheduled the task. **But this only worked because we explicitly
passed `num_gpus=1` to `ray.init()`.**

### CRITICAL FINDING: Ray does NOT auto-detect Intel XPU as a GPU resource

```python
ray.init()  # no explicit num_gpus
print(ray.cluster_resources())
# -> {'node:...': 1.0, 'memory': ..., 'object_store_memory': ..., 'CPU': 20.0}
#    NOTE: no 'GPU' key at all
```

Ray's automatic GPU detection is built around NVIDIA-specific mechanisms
(`nvidia-smi`/CUDA) and has no native Intel XPU/Level-Zero detection path. When
`train_ppo_ray.py` calls `ray.init()` (its actual code, no `num_gpus` argument — see
`train_ppo_ray.py` line 23), Ray silently detects **zero GPUs**, and any `@ray.remote(num_gpus=1)`
actor request then hangs **indefinitely** waiting for GPU capacity Ray doesn't think exists —
even though the Arc B70 is physically present and otherwise working.

A teammate found Ray's own documentation confirms this is expected, not a bug:
Intel GPU support is listed under Ray's accelerator docs as experimental/community-supported,
and the documented pattern requires **explicitly declaring** GPU resources at cluster
startup — Ray does not infer them.
(https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)

### The fix — start Ray's cluster externally, with GPUs declared, before running the training script

Three options were evaluated:
1. **Custom resource** (`ray.init(resources={"XPU": 1})`) — rejected: would require rewriting
   every `@ray.remote(num_gpus=1)` decorator across `ppo_actor.py`, `ppo_critic.py`,
   `launcher.py` to a custom key, breaking compatibility with the same code running on NVIDIA.
2. **Add `num_gpus=` directly inside `train_ppo_ray.py`'s own `ray.init()` call** — rejected:
   requires editing OpenRLHF's core `ray.init()` logic, and hardcoding a GPU count there is
   wrong on a multi-GPU node.
3. **Start Ray as an external cluster first, with GPUs declared via the CLI, then let
   `train_ppo_ray.py` attach to it** — **adopted.** `train_ppo_ray.py` already checks
   `if not ray.is_initialized():` before calling `ray.init()`. If a cluster is already
   running, it just attaches — zero code changes needed.

```bash
ray stop --force
rm -rf /tmp/ray
NUM_GPUS=$(python -c "import torch; print(torch.xpu.device_count())")
ONEAPI_DEVICE_SELECTOR=level_zero:0 ray start --head --num-gpus="$NUM_GPUS"
```
`NUM_GPUS` is read from `torch.xpu.device_count()` rather than hardcoded — this must match the
actual GPU count on whichever machine this runs on. Passing a number lower than the real count
just leaves extra GPUs idle; passing a number higher than the real count makes Ray schedule
multiple jobs onto the same physical GPU, causing silent contention/OOM rather than a clear
error — always detect it, don't hardcode it.

Verified: after this, `ray.cluster_resources()` correctly shows `'GPU': 1.0` (this machine has
1 GPU), and GPU-scheduled actors (`PolicyModelActor`, `ReferenceModelActor`, etc.) were placed
and started successfully.

**This must be run before every `train_ppo_ray.py` invocation** (or wired into a startup
script) until/unless Ray adds native Intel GPU auto-detection.

---

## Part 2 — PPO (`train_ppo_ray.py`): 7 real bugs found and fixed, 1 architectural blocker remains

### Bug 1 — unconditional `vllm` import blocks `train_ppo_ray --help` entirely
Same class of bug as the `flash_attn` import fix in `ring_attn_utils.py`. `openrlhf/trainer/ray/__init__.py`
did `from .vllm_engine import batch_vllm_engine_call, create_vllm_engines` at module load
time; `vllm_engine.py` does `import vllm` at its top. Since `vllm` cannot be installed on
this XPU machine via plain pip (see prior findings — pulls in ~15 NVIDIA CUDA packages and
would downgrade torch), this crashed the CLI before printing any help text, even though
`--vllm.num_engines 0` is a documented, supported way to disable vLLM entirely.

**Fix:** removed the eager import from `trainer/ray/__init__.py`; moved
`from openrlhf.trainer.ray import create_vllm_engines` inside `train_ppo_ray.py`'s
`if args.vllm.num_engines is not None and args.vllm.num_engines > 0:` block (its only caller).

### Bugs 2-4 — same unconditional-import pattern, 3 more files
Running the actual training (not just `--help`) surfaced the identical bug pattern in three
more places, each fixed the same way (import moved inside the function/branch that actually
uses it):
- `openrlhf/trainer/ppo_utils/samples_generator.py` — top-level `from vllm import SamplingParams`
  and `from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call` (4 call sites,
  all already guarded by `if self.args.vllm.enable_sleep:` — just needed the import moved inside).
- `openrlhf/trainer/ppo_trainer.py` — top-level `from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call`,
  used only inside an `if self.args.vllm.enable_sleep:` block.
- `openrlhf/trainer/ppo_trainer_async.py` — same pattern (fixed for consistency, though this
  file is only reached with `--train.async_enable`, not exercised in this test).

### Bug 5 — `sync_backend` guard doesn't check if vLLM is actually enabled
`openrlhf/trainer/ray/ppo_actor.py`, `init_model_from_pretrained()`:
```python
# Before:
if getattr(args.vllm, "sync_backend", "nccl") == "nccl":
    import vllm
    ...
# After:
if vllm_engines and getattr(args.vllm, "sync_backend", "nccl") == "nccl":
    import vllm
    ...
```
`sync_backend` defaults to `"nccl"` regardless of whether vLLM is enabled at all — so this
branch (and its `import vllm`) executed even with `--vllm.num_engines 0` / `vllm_engines=None`.
Added a check that `vllm_engines` is actually truthy first.

### Bug 6 — `max_len` UnboundLocalError with `--vllm.num_engines 0` (NOT an XPU bug — affects any platform)
`openrlhf/cli/train_ppo_ray.py`:
```python
# Before:
if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
    max_len = args.data.max_len   # only defined in this branch
    ...
...
ppo_trainer = PPOTrainer.remote(..., max_len=max_len, ...)   # used unconditionally, later

# After:
max_len = args.data.max_len   # moved outside the vllm-conditional block
if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
    ...
```
This is a genuine OpenRLHF logic bug, independent of XPU/CUDA — `max_len` was only ever
assigned inside the vLLM-enabled branch, but read unconditionally later. Would crash
identically on an NVIDIA machine running with `--vllm.num_engines 0`.

### Bug 7 — `--train.colocate_all` alone cannot fit 4 PPO roles on 1 GPU (architectural, not a bug per se)
With `actor`, `ref`, `critic`, `reward` all requesting `num_gpus_per_node=1`, plus
`--train.colocate_all --train.colocate_actor_ref --train.colocate_critic_reward`, the run
hung indefinitely (even with Ray's GPU auto-detection fixed). Root cause traced in
`train_ppo_ray.py`: `colocate_actor_ref` creates **one** placement group (1 GPU bundle) for
actor+ref; `colocate_critic_reward` creates a **second, separate** placement group (another
1 GPU bundle) for critic+reward. Actors inside a placement group request only a fractional
GPU share (`num_gpus_per_actor=0.2`), but each placement group still reserves a **whole**
bundle. Two placement groups × 1 GPU bundle each = 2 GPUs required minimum — this
configuration cannot fit on a single physical GPU regardless of platform.

**Workaround used:** switched `--algo.advantage.estimator` to `group_norm` (GRPO-style,
group-normalized advantages). `train_ppo_ray.py` line ~601 auto-disables the critic entirely
when the estimator isn't `"gae"` — removing the second placement group, fitting everything
into actor+ref's single 1-GPU bundle (with reward model also sharing via `colocate_all`'s
first `pg`).

### Result after all 7 fixes: real progress, but a final architectural blocker

With all fixes applied, Ray cluster GPU-declared, and `group_norm` avoiding the 2-GPU
requirement:
- Dataset loaded (`OpenRLHF/prompt-collection-v0.1`, 179465 rows, filtered to 4 samples)
- `PolicyModelActor`, `ReferenceModelActor`, `RewardModelActor` all initialized successfully
- Qwen2.5-0.5B model loaded into all three actors
- DeepSpeed's `FusedAdam` JIT-compiled and loaded (confirms the icpx/compiler fix from the
  SFT investigation carries over correctly to the PPO path)
- Training loop started: `Episode [1/1]: 0/4`

Then hit the final blocker:
```
File "openrlhf/trainer/ppo_utils/samples_generator.py", line 215, in _dispatch_prompts_to_vllm
    from vllm import SamplingParams
ModuleNotFoundError: No module named 'vllm'
```

### FINAL CONCLUSION: this is not a bug — it's a genuine architectural requirement

Unlike bugs 1-6, this import IS correctly placed (lazy, inside the function that uses it) —
we placed it there ourselves in Bug 2's fix. The problem is that **this function is actually
being called**. Checked `SamplesGenerator`'s available methods:
```
generate_eval_samples()
generate_samples()
_generate_vllm()      <- the only rollout-generation method that exists
```

**There is no non-vLLM sample-generation path in this version of OpenRLHF's PPO trainer.**
`--vllm.num_engines 0` only skips *spinning up dedicated vLLM engine processes* — the actual
rollout/text-generation step (`generate_samples()` → `_generate_vllm()` →
`_dispatch_prompts_to_vllm()`) unconditionally calls into vLLM's `SamplingParams` API
regardless. This means: **PPO/GRPO training in this OpenRLHF version requires the `vllm`
package to be importable, full stop — not just for performance, but because there is no
alternative code path implemented for generating rollouts via plain DeepSpeed/HF generate().**

This is consistent with, and now fully explains, the vLLM investigation from Phase-1's
dependency findings: vLLM cannot install via plain pip on Intel XPU (would corrupt the
torch+xpu stack), and the real path requires building vLLM from source with
`VLLM_TARGET_DEVICE=xpu` (deferred, per `MASTER_setup_and_fixes_guide.md` Part 5).
**PPO/GRPO cannot be smoke-tested end-to-end on this machine until that vLLM-from-source
build is completed** — this is not something further OpenRLHF code fixes can work around,
short of implementing a whole new non-vLLM generation path (a much larger undertaking, and
not an XPU-specific fix — it would be a genuine new feature for OpenRLHF itself).

---

## Summary table

| Finding | Category | Status |
|---|---|---|
| Ray core (init, remote tasks, XPU visibility from workers) | Platform | ✅ Confirmed working |
| Ray GPU auto-detection doesn't see Intel XPU | Platform limitation (Ray, not OpenRLHF) | ✅ Worked around (`ray start --head --num-gpus=N`) |
| 5 unconditional `vllm` imports across the PPO codebase | OpenRLHF code bug | ✅ Fixed (lazy-import pattern) |
| `sync_backend` guard missing `vllm_engines` check | OpenRLHF code bug | ✅ Fixed |
| `max_len` UnboundLocalError with `--vllm.num_engines 0` | OpenRLHF code bug (platform-independent) | ✅ Fixed |
| `colocate_all` needs 2 GPUs minimum for actor+ref+critic+reward | Architectural constraint, not a bug | ✅ Worked around (`group_norm` estimator, no critic) |
| Rollout generation has no non-vLLM fallback | Architectural requirement | ❌ **Not fixable without vLLM** — blocked pending vLLM-from-source build |

## Where things are saved
- This report: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/ray_and_ppo_findings.md`
- Code changes: `openrlhf/trainer/ray/__init__.py`, `openrlhf/cli/train_ppo_ray.py`,
  `openrlhf/trainer/ppo_utils/samples_generator.py`, `openrlhf/trainer/ppo_trainer.py`,
  `openrlhf/trainer/ppo_trainer_async.py`, `openrlhf/trainer/ray/ppo_actor.py`,
  `openrlhf/trainer/ray/ppo_critic.py`, `openrlhf/trainer/ray/launcher.py`,
  `openrlhf/trainer/ppo_utils/replay_buffer.py`
- Raw logs: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/logs/ppo_smoke_test*.log`
