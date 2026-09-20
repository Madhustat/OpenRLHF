# Code review: XPU device pinning in `vllm_engine.py`

**File:** `openrlhf/trainer/ray/vllm_engine.py` (1 production file). Test file
`tests/test_vllm_device_env.py` updated to match; not imported by training.

**Baseline for comparison:** upstream OpenRLHF (`OpenRLHF-baseline`), before any XPU work.

---

## Problem statement

`RolloutRayActor` pins each vLLM engine to the accelerator Ray assigned it. Two defects on
Intel XPU:

1. **Wrong variable.** Upstream sets only `CUDA_VISIBLE_DEVICES`. vLLM's XPU platform reads
   `ZE_AFFINITY_MASK`, not `CUDA_VISIBLE_DEVICES`, so the pin had no effect on XPU.
2. **Wrong timing.** `import vllm` (and `from openrlhf.utils.agent import ...`) ran at module
   top. Importing vllm initializes the accelerator runtime (Intel Level-Zero), which enumerates
   and **locks device visibility**. Ray imports this module inside each actor *before*
   `__init__` runs, so visibility was frozen before the per-actor mask was set — every actor
   fell back to device 0 on a multi-GPU host.

Both are silent: single-GPU runs are unaffected (only device 0 exists), and the original test
checked only the env-var string, so it passed on a real collision.

---

## Verification of the timing claim (decisive test)

`torch.xpu.is_initialized()` is an unreliable detector here. The reliable test sets an
out-of-range mask and checks whether device enumeration still honors it:

| Order | `torch.xpu.device_count()` | Meaning |
|---|---|---|
| set mask, then `import vllm` | reflects mask | visibility still malleable — OK |
| `import vllm`, then set mask | mask **ignored** (count unchanged) | runtime already locked — too late |
| touch `vllm.platforms.current_platform`, then set mask | mask **ignored** | reading the platform also locks it |

Confirmed on the single-XPU box. This is why the fix must (a) defer the vllm import until
after pinning, and (b) read the control-var name from `torch.version` rather than
`vllm.platforms`.

---

## Change 1 — Defer runtime-initializing imports

**Before**
```python
import vllm
from vllm.inputs import TokensPrompt
from vllm.utils import random_uuid
from openrlhf.utils.agent import AgentExecutorBase, SingleTurnAgentExecutor
```

**After** — removed from module top; imported locally where used:
- `__init__`: `import vllm` after `self._configure_device_env(...)`
- `__init__`: `from openrlhf.utils.agent import SingleTurnAgentExecutor`
- `_load_agent_executor`: `from openrlhf.utils.agent import AgentExecutorBase`
- `generate`: `from vllm.inputs import TokensPrompt` / `from vllm.utils import random_uuid`
- `create_vllm_engines` (logprobs branch): `import vllm`

**Why required:** Ray imports the module in each actor before `__init__`. Any top-level vllm/agent
import locks device visibility before the mask is set. Deferring guarantees pinning happens
first. **NVIDIA impact: none** — Ray sets `CUDA_VISIBLE_DEVICES` before the worker process
starts, so import order inside the process is irrelevant on CUDA.

**Verified:** after the change, importing the module leaves `vllm` and `deepspeed` out of
`sys.modules`.

---

## Change 2 — Resolve the control var from `torch.version`, not `vllm.platforms`

**Added**
```python
def _resolve_device_control_env_var():
    import torch
    if getattr(torch.version, "xpu", None):
        return "ZE_AFFINITY_MASK"
    if getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
        return "CUDA_VISIBLE_DEVICES"
    return None
```

**Why required:** we need the platform's control-var name inside `_configure_device_env`, which
runs before pinning. The obvious source, `vllm.platforms.current_platform.device_control_env_var`,
initializes the runtime just by being read (verified above) — the chicken-and-egg problem.
`torch.version.{xpu,cuda,hip}` is readable without initializing the runtime and reproduces
vLLM's mapping exactly (verified: both return `ZE_AFFINITY_MASK` on this box).

**Trade-off (reviewer note):** this reimplements a mapping vLLM owns. Accepted because the
alternative (reading `current_platform`) defeats the timing fix. `_configure_vllm_env` still
uses vLLM after import, so this is only for the pre-import window.

---

## Change 3 — Set the platform var (not just CUDA), with guards

**Before**
```python
elif ray_noset_visible_devices():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])
```

**After**
```python
elif ray_noset_visible_devices():
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("Ray did not assign an accelerator to the vLLM actor")
    if not device_control_env_var:
        raise RuntimeError("The active vLLM platform does not define a device-control environment variable")
    if device_control_env_var == "ZE_AFFINITY_MASK":
        os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
    os.environ[device_control_env_var] = str(gpu_ids[0])
```

**Why required:**
- Sets `ZE_AFFINITY_MASK` on XPU (the variable vLLM actually reads); still
  `CUDA_VISIBLE_DEVICES` on NVIDIA (unchanged behavior).
- `ONEAPI_DEVICE_SELECTOR` and `ZE_AFFINITY_MASK` intersect in Level-Zero; if Ray set the
  selector, keeping it alongside the mask can yield 0 visible devices. Cleared before masking,
  XPU-only.
- Empty `ray.get_gpu_ids()` now raises a clear error instead of an opaque `IndexError`.

---

## Change 4 — `ray` branch also clears the platform var

**Before:** popped `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`.
**After:** the loop also pops `device_control_env_var` (adds `ZE_AFFINITY_MASK` on XPU).

**Why required:** when `distributed_executor_backend == "ray"`, vLLM's ray executor assigns
devices to its child workers; a stale inherited mask must be cleared. On NVIDIA this is a
no-op (`device_control_env_var` is `CUDA_VISIBLE_DEVICES`, already in the set).

---

## What was intentionally NOT added

- `_assert_runtime_not_initialized` guard and `_verify_vllm_device_mapping` cross-check were
  proposed but dropped to keep the change minimal. The deferred imports are the fix; the guard
  only polices them.

---

## Validation

| Check | Result |
|---|---|
| Module import does not load vllm/deepspeed | PASS (`sys.modules` clean) |
| Decisive timing test (mask honored only pre-import) | PASS |
| `_resolve_device_control_env_var()` == vLLM mapping | PASS (`ZE_AFFINITY_MASK`) |
| Unit tests `tests/test_vllm_device_env.py` | 5/5 PASS |
| E2E GRPO, single XPU | EXIT 0, rewards matching |

**Still owed (needs hardware):** multi-XPU physical UUID proof (two engines → two distinct
devices) and a real NVIDIA run confirming the CUDA path is unchanged. The multi-GPU collision
is only observable on a >=2-GPU host.

---

## NVIDIA-safety summary (for reviewers on the CUDA path)

- `_resolve_device_control_env_var()` returns `CUDA_VISIBLE_DEVICES` on CUDA — same variable as
  upstream.
- Deferred imports don't affect CUDA: Ray sets `CUDA_VISIBLE_DEVICES` before the worker starts,
  so in-process import order is irrelevant.
- `ONEAPI_DEVICE_SELECTOR` pop is guarded by `== "ZE_AFFINITY_MASK"` — never runs on CUDA.
- `ray`-branch addition is a no-op on CUDA (the added var is already `CUDA_VISIBLE_DEVICES`).
