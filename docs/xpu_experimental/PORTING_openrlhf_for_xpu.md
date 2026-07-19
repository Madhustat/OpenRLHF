# Porting OpenRLHF to Intel XPU — Source Code Changes

This document lists every change made to OpenRLHF's own source code to make it run correctly
on Intel XPU hardware, verified directly against `git diff origin/main` on 2026-07-15 — not
reconstructed from memory or from an earlier, incomplete draft. Apply these to a fresh
`git clone` of OpenRLHF. For environment/dependency setup (not code changes), see the
companion document `SETUP_new_machine_from_scratch.md`.

**Underlying pattern:** almost every fix below follows the same shape. OpenRLHF's original
code assumes CUDA is the only possible accelerator and calls `torch.cuda.*` directly, or
hardcodes the string `"cuda"`. These calls either don't exist or silently do the wrong thing on
an XPU-only torch build. The fix is consistently PyTorch's own vendor-neutral
`torch.accelerator.*` API, which resolves to `"cuda"` on NVIDIA machines (fully backward
compatible, zero behavior change there) and `"xpu"` here. A few changes are a different shape
(optional-dependency import ordering, a broadcast-communicator fallback) — those are called out
explicitly.

**18 files modified, 2 new test files added.** Nothing in this list is a workaround that needs
to be reverted later — every change here is either a genuine bug fix (works correctly on CUDA
too) or a necessary device-generic replacement.

---

## 1. `openrlhf/models/ring_attn_utils.py`

**Why this file needs changes at all:** it unconditionally imports from the `flash_attn` PyPI
package at the top of the file. `flash_attn` cannot be installed on Intel XPU — it compiles
CUDA kernels via `nvcc`, and there's no CUDA toolchain here. This import failing blocks
`openrlhf.models` entirely — even `train_sft --help` crashes before printing anything,
regardless of whether the run would ever actually use ring attention.

**Change 1 — imports:**
```python
# BEFORE:
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather

# AFTER:
from einops import rearrange
from transformers.modeling_flash_attention_utils import (
    _index_first_axis as index_first_axis,
    _pad_input as pad_input,
    _unpad_input as unpad_input,
)
```
`rearrange` moves to its true original source (`einops` — `flash_attn` was only re-exporting
it). `index_first_axis`/`pad_input`/`unpad_input` now come from `transformers` (already an
OpenRLHF dependency) instead of `flash_attn`. Verified signature-compatible.

**Change 2 — vendored `all_gather` replacement** (transformers has no equivalent for this one):
```python
class _AllGatherFunc(torch.autograd.Function):
    """Minimal replacement for flash_attn.utils.distributed.all_gather.
    Pure torch.distributed, no CUDA-specific code."""

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        world_size = dist.get_world_size(process_group)
        output = torch.empty(world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device)
        dist.all_gather_into_tensor(output, input_.contiguous(), group=process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size(ctx.process_group)
        grad_input = torch.empty(
            grad_output.shape[0] // world_size, *grad_output.shape[1:], dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(grad_input, grad_output.contiguous(), group=ctx.process_group)
        return grad_input, None


all_gather = _AllGatherFunc.apply
```
Uses only `torch.distributed` primitives (`all_gather_into_tensor`, `reduce_scatter_tensor`) —
zero CUDA-specific code, verified by reading flash-attn's actual source for this function
before writing the replacement.

**Change 3 — hardcoded CUDA device**, inside `reset_ring_attn_position_ids()`:
```python
# BEFORE:
position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
# AFTER:
position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.accelerator.current_accelerator())
```
`torch.cuda.current_device()` raises `AssertionError: Torch not compiled with CUDA enabled` on
an XPU-only build.

---

## 2. `openrlhf/utils/deepspeed/deepspeed.py`

OpenRLHF's DeepSpeed integration layer — every training mode (SFT/RM/DPO/PPO) goes through
`DeepspeedStrategy`. Confirmed separately that DeepSpeed itself correctly reports `"xpu"` via
`deepspeed.accelerator.get_accelerator()` — this file bypassing that abstraction with direct
`torch.cuda.*` calls is the actual problem, not DeepSpeed itself.

| Location | Before | After |
|---|---|---|
| `setup_distributed()`, local_rank | `torch.cuda.set_device(local_rank)` | `torch.accelerator.set_device_index(local_rank)` |
| `setup_distributed()`, device mesh | `init_device_mesh("cuda", (dp_size, ring_attn_size, tp_size), mesh_dim_names=("dp","sp","tp"))` | `init_device_mesh(torch.accelerator.current_accelerator().type, (dp_size, ring_attn_size, tp_size), mesh_dim_names=("dp","sp","tp"))` |
| tensor-parallel init path | `gc.collect(); torch.cuda.empty_cache()` | `torch.accelerator.empty_cache()` |
| `all_reduce()` method | `data.to(torch.cuda.current_device())` | `data.to(torch.accelerator.current_accelerator())` |
| `all_gather()` method (×2) | `torch.zeros_like(data).to(torch.cuda.current_device())` / `data.to(torch.cuda.current_device())` | same pattern with `torch.accelerator.current_accelerator()` |

**Symptom without the device-mesh fix specifically:**
`AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'` — this symbol simply
doesn't exist in a torch build compiled for XPU.

**Symptom without the tensor-parallel `empty_cache()` fix — worth flagging as the most
dangerous one in this file:** no crash at all. `torch.cuda.empty_cache()` silently does nothing
on an XPU-only torch build (it doesn't error, it just no-ops), which causes unreleased GPU
memory to build up silently — risking a confusing, delayed OOM much later instead of an
immediate, traceable error at the actual point of the bug.

---

## 3. `openrlhf/utils/loss_utils.py`

```python
# BEFORE:
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# AFTER:
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
```
**Why this one is worse than a simple crash:** the old fallback check only asks "is CUDA
available?" — on XPU this correctly evaluates `False`, so it silently falls through to `"cpu"`
instead of `"xpu"`. Tensors then get created on CPU, and later code calls `dist.all_reduce()`
on them over the `xccl` process group — which has **no CPU-communication path registered** —
producing a confusing `RuntimeError: No backend type associated with device type cpu`, nowhere
near the actual bug (a fallback that should have picked `xpu`, not `cpu`).

---

## 4. `openrlhf/utils/distributed_util.py`

**Change 1:**
```python
# BEFORE:
def torch_dist_barrier_and_cuda_sync():
    torch.distributed.barrier()
    torch.cuda.synchronize()
# AFTER:
def torch_dist_barrier_and_cuda_sync():
    torch.distributed.barrier()
    torch.accelerator.synchronize()
```
Hit at the end of a training run, inside `strategy.save_model()`'s checkpoint-save cleanup
barrier: `AssertionError: Torch not compiled with CUDA enabled`.

**Change 2 — a real functional bug, not just a portability fix.** `stateless_init_process_group`
originally always used `PyNcclCommunicator` for weight-sync broadcasts (DeepSpeed actor → vLLM
engine). On Intel XPU, **`PyNcclCommunicator` silently sets `disabled=True`** when no NCCL/RCCL
is present — every `broadcast()` call becomes a silent no-op. This was confirmed directly with
live instrumentation: the code appeared to run successfully (no exception, no error), but
weight sync from the trained actor to vLLM's inference engine was never actually happening.

Fix — added a new class, and branch on whether NCCL/RCCL is actually present:
```python
class _GlooBroadcastCommunicator:
    """Wraps a real gloo ProcessGroup to expose the same .broadcast(tensor, src) -> tensor
    interface as PyNcclCommunicator. Unlike PyNcclCommunicator, this returns a NEW tensor
    rather than mutating in place, and stages through CPU (gloo doesn't support XPU tensors
    directly): tensor.cpu() -> pg.broadcast() -> .to(self.device)."""
    # ... (see actual file for full implementation)


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    if torch.version.cuda is not None or torch.version.hip is not None:
        # original NCCL/RCCL path, unchanged
        pg = StatelessProcessGroup.create(...)
        return PyNcclCommunicator(pg, device=device)
    else:
        # Intel XPU (no NCCL/RCCL): real gloo collective instead of a silently-disabled NCCL stub
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group
        pg = stateless_init_torch_distributed_process_group(
            master_address, master_port, rank, world_size, backend="gloo"
        )
        return _GlooBroadcastCommunicator(pg, device)
```
**Performance note, in case a "faster" alternative is tried later:** an earlier attempt used
`StatelessProcessGroup`'s own TCP-store-based broadcast directly (no gloo). It was functionally
correct but far slower under GPU colocation — measured at ~2.3s per parameter (vs. ~3-4ms in
isolation), totaling ~809 seconds for one weight-sync round on a 0.5B model. The real-gloo-
collective approach above is 540-700x faster in practice. Don't regress to the TCP-store
approach without re-measuring.

---

## 5. `openrlhf/utils/deepspeed/deepspeed_utils.py`

```python
# offload_deepspeed_states():
torch.cuda.empty_cache()   -> torch.accelerator.empty_cache()
torch.cuda.synchronize()   -> torch.accelerator.synchronize()

# reload_deepspeed_states():
torch.cuda.empty_cache()   -> torch.accelerator.empty_cache()
torch.cuda.synchronize()   -> torch.accelerator.synchronize()
```
These functions back `--ds.enable_sleep` (requires `--ds.zero_stage >= 1`). Not exercised until
this exact feature combination was actually tested — `AssertionError: Torch not compiled with
CUDA enabled` before the fix.

---

## 6. `openrlhf/trainer/ppo_utils/replay_buffer.py`

```python
# BEFORE:
self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
# AFTER:
self.target_device = torch.accelerator.current_accelerator()
```
Plus three more occurrences of `torch.tensor(..., device=torch.cuda.current_device())` →
`device=torch.accelerator.current_accelerator()` (for `num_steps`, `num_microbatches`,
`global_valid_sample_num`/`global_num_tokens`).

---

## 7. `openrlhf/trainer/ray/launcher.py`

```python
# BaseModelActor.empty_cache():
torch.cuda.empty_cache(); torch.cuda.synchronize()
# ->
torch.accelerator.empty_cache(); torch.accelerator.synchronize()
```
Plus two occurrences of `device = torch.cuda.current_device()` (in `ReferenceModelActor` and
`RewardModelActor` forward methods) → `torch.accelerator.current_accelerator()`.

---

## 8. `openrlhf/trainer/ray/ppo_actor.py`

Multiple `torch.cuda.*` → `torch.accelerator.*` substitutions:
- `stateless_init_process_group(..., torch.cuda.current_device())` → `torch.accelerator.current_accelerator()`
- `device = torch.cuda.current_device()` in the training loop and forward-value method → `torch.accelerator.current_accelerator()`
- `torch.cuda.empty_cache()` (multiple call sites in `broadcast_to_vllm()` and `fit()`) → `torch.accelerator.empty_cache()`
- `torch.cuda.synchronize()` in `fit()` → `torch.accelerator.synchronize()`

**Broadcast call signature change** (works together with the `distributed_util.py` fix above):
```python
# BEFORE:
self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
# AFTER:
param.data = self._model_update_group.broadcast(param.data, src=0)
```
The `stream=` kwarg is CUDA-specific and doesn't exist on the gloo-based fallback communicator;
the gloo path also returns a new tensor rather than mutating in place, so the return value must
be captured.

**Guard fix** — prevents inspecting vLLM's version/attributes when no vLLM engines exist at all:
```python
# BEFORE:
if getattr(args.vllm, "sync_backend", "nccl") == "nccl":
# AFTER:
if vllm_engines and getattr(args.vllm, "sync_backend", "nccl") == "nccl":
```

---

## 9. `openrlhf/trainer/ray/ppo_critic.py`

Same pattern as `ppo_actor.py`: `device = torch.cuda.current_device()` (train loop and
forward/value method) → `torch.accelerator.current_accelerator()`; `torch.cuda.empty_cache()`
(×2) / `torch.cuda.synchronize()` (×1) in `fit()` → `torch.accelerator.*` equivalents.

---

## 10. `openrlhf/trainer/ray/utils.py`

```python
# get_physical_gpu_id() — BEFORE:
device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

# AFTER:
backend = getattr(torch, torch.accelerator.current_accelerator().type)
device = backend.current_device()
props = backend.get_device_properties(device)
```
**Why the indirection through `getattr`:** `torch.accelerator` has no generic
`get_device_properties()` method of its own — you have to resolve the actual backend module
(`torch.cuda` or `torch.xpu`) by name first, then call its device-specific method on that
module.

---

## 11. `openrlhf/trainer/ray/vllm_worker_wrap.py`

```python
# BEFORE:
weight = torch.empty(shape, dtype=dtype, device="cuda")
...
self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

# AFTER:
weight = torch.empty(shape, dtype=dtype, device=torch.accelerator.current_accelerator())
...
weight = self._model_update_group.broadcast(weight, src=0)
```
Same reasoning as `ppo_actor.py`'s broadcast fix — hardcoded `device="cuda"` fails outright on
XPU-only hardware; `stream=` is dropped for the gloo fallback path.

**Left deliberately unfixed:** `update_weight_cuda_ipc()` in this same file, which uses CUDA
Inter-Process Communication (`torch.multiprocessing.reductions.reduce_tensor`) — a
memory-sharing mechanism fundamentally tied to CUDA's driver model. vLLM does ship an
`xpu_communicator.py` suggesting an XPU-native equivalent exists, but it was not needed: passing
`--vllm.sync_backend gloo` (see below) avoids this code path entirely by keeping
`use_cuda_ipc` false.

**Why `--vllm.sync_backend gloo` matters here:** `ppo_actor.py`'s
`self.use_cuda_ipc = backend == "nccl" and self.args.train.colocate_all and not self.args.train.async_enable`
evaluates `True` with the default `sync_backend="nccl"` plus a single-GPU `--train.colocate_all`
config, routing weight-sync through the unfixed CUDA-IPC path. Passing
`--vllm.sync_backend gloo` makes this `False`, routing through the fixed `update_weight()` path
instead. This is a legitimate configuration choice, not a hack — `gloo` is a real,
general-purpose PyTorch distributed backend, and using it here only changes *how* weights get
copied from DeepSpeed into vLLM, not any actual training math.

---

## 12. Lazy-import fixes (5 files) — optional `vllm` dependency

`--vllm.num_engines 0` is a valid, supported configuration (DeepSpeed-only rollout, no vLLM at
all) — but several files imported `vllm`-dependent symbols eagerly at module load time, which
crashed immediately even when vLLM was never going to be used.

| File | Before | After |
|---|---|---|
| `openrlhf/trainer/ray/__init__.py` | `from .vllm_engine import batch_vllm_engine_call, create_vllm_engines` + `__all__` | Removed; replaced with a comment instructing callers to import directly from `.vllm_engine` at point of use |
| `openrlhf/cli/train_ppo_ray.py` | `from openrlhf.trainer.ray import create_vllm_engines` (top-level) | `from openrlhf.trainer.ray.vllm_engine import create_vllm_engines` as a lazy import inside `train()`, only reached when `args.vllm.num_engines > 0` |
| `openrlhf/trainer/ppo_trainer.py` | `from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call` (top-level) | Moved inside the `if self.args.vllm.enable_sleep:` block |
| `openrlhf/trainer/ppo_trainer_async.py` | Same top-level import | Moved inside `broadcast_to_vllm()` |
| `openrlhf/trainer/ppo_utils/samples_generator.py` | `from vllm import SamplingParams` + `from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call` (top-level) | Both moved inside the specific functions that use them (wake_up/sleep blocks, prompt-dispatch method) |

**Note on `train_ppo_ray.py`'s import specifically:** this needed a follow-up fix after the
`ray/__init__.py` change above — `train_ppo_ray.py` still imported from the **package**
(`openrlhf.trainer.ray`) rather than the submodule, which raised
`ImportError: cannot import name 'create_vllm_engines' from 'openrlhf.trainer.ray'` the moment
`--vllm.num_engines > 0` actually needed it. Both parts of this fix are required together.

**Keep these permanently, even now that vLLM works:** they're harmless once vLLM is installed
(import still succeeds, just resolved a few lines later) and reverting them reintroduces a real
crash for anyone using `--vllm.num_engines 0`.

---

## 13. `openrlhf/cli/train_rm.py` — unrelated small-dataset fix

```python
# BEFORE:
eval_data = train_data.select(range(min(args.data.max_samples, int(len(train_data) * 0.01))))
# AFTER:
eval_data = train_data.select(range(min(args.data.max_samples, max(1, int(len(train_data) * 0.01)))))
```
**Not a CUDA/XPU fix** — flagging this explicitly so it isn't mistaken for a device-API port.
`int(n * 0.01)` rounds to 0 for small datasets (common when smoke-testing with a handful of
samples), producing an empty eval slice that crashes downstream dataset mapping. Floors the
eval slice at 1 row. Needed for testing on this hardware with small sample counts, but not
XPU-specific in cause.

---

## 14. Test suite additions

Two new files added; no existing test files were modified in a way that changes their
behavior for CUDA (only `test_ray_env_vars.py` gained 2 new test functions, covered below).
**Two previously-existing XPU-specific test files
(`test_xccl_backend.py`, `test_loss_aggregation_xpu.py`) were deliberately removed** — see the
note at the end of this section for why, and what replaced them.

### 14.1 `tests/test_loss_aggregation_generic.py` (new, 8 tests)

Device-generic version of the original `test_loss_aggregation.py` tests. The original file
creates tensors with no explicit `device=` argument, which defaults to CPU — so it only ever
exercised the CPU code path, never the accelerator-resident path real training actually uses.
This file parametrizes the same 8 test cases over `["cpu"] + [detected accelerator]`, using
`torch.accelerator` for detection — no hardcoded vendor list, so it also covers CUDA or ROCm
automatically if run on that hardware. `test_loss_aggregation.py` itself is left unmodified.

### 14.2 `tests/test_distributed_backend_generic.py` (new, 2 tests)

Verifies real distributed collective communication works, generically across whatever backend
is canonical for the detected device (`nccl`/cuda, `xccl`/xpu, `gloo`/cpu — resolved via
`torch.distributed.get_default_backend_for_device()`, no hardcoded vendor branching):

- `test_backend_is_available` — confirms the canonical backend for each detected device is
  actually compiled into the installed torch build (a build/packaging check, distinct from
  whether the hardware itself works).
- `test_two_rank_allreduce_crosses_processes` — spawns two **real, separate OS processes** via
  `torch.multiprocessing`, each with a different starting value, and asserts both end up with
  the correctly-summed result after `all_reduce`. This specifically proves data crosses
  process boundaries — a single-process (`world_size=1`) collective call would trivially
  return its own input unchanged and could pass even if inter-process communication were
  completely broken, so this test always uses two real ranks. Device assignment scales via
  `rank % device_count`, so the same code works on single-GPU (both ranks share device 0) and
  multi-GPU (each rank gets its own device) hardware without modification.

### 14.3 `tests/test_ray_env_vars.py` (2 new tests added to existing file)

`CCL_LOG_LEVEL` is oneCCL/XCCL's own logging-verbosity environment variable — confirmed via
oneCCL's own documentation that `NCCL_DEBUG` (NVIDIA-only) has **no effect** on it; they are
separate variables read by separate libraries, with different accepted value casing
(`"WARN"` for NCCL, `"warn"` for oneCCL). Before this fix, only `NCCL_DEBUG` was set in
`train_ppo_ray.py`'s `ray.init()` call — meaning XPU runs had no working log-verbosity
default at all for the library actually in use. Fix: `CCL_LOG_LEVEL` is now set unconditionally
alongside `NCCL_DEBUG`, each library reads only its own variable and ignores the other, so both
NVIDIA and Intel machines get a working default regardless of which hardware is present. Two
new tests mirror the existing `NCCL_DEBUG` test pattern in this file: `test_ccl_log_level_default`
(unset → defaults to `"warn"`) and `test_ccl_log_level_user_debug` (user override is respected,
not silently discarded).

### 14.4 Removed: `test_xccl_backend.py`, `test_loss_aggregation_xpu.py`

These were earlier, XPU-hardcoded test files (not upstream — written during this project).
`test_loss_aggregation_xpu.py`'s coverage is now provided by the generic parametrized version
(14.1). `test_xccl_backend.py` was a regression guard for one specific historical bug (`ze_data
was not initialized`, see `SETUP_new_machine_from_scratch.md` Part 1.1) — removed in favor of
relying on that setup step being documented, rather than keeping a dedicated test for it. If
you want that regression guard back, its logic is straightforward to reconstruct: real
`init_process_group(backend="xccl")` + `all_reduce()` on a real XPU tensor, single process is
sufficient since the failure occurs at initialization, before rank count matters.

---

## Current test suite state (2026-07-15)

```bash
cd OpenRLHF && python -m pytest tests/ -v
```
**4 files, 39 tests, 39/39 passing:**

| File | # tests | Covers |
|---|---|---|
| `test_loss_aggregation.py` | 8 | Original upstream CPU-only loss-aggregation math (unmodified) |
| `test_ray_env_vars.py` | 11 (9 original + 2 new) | Ray `runtime_env` env-var defaulting/override logic (mocked Ray, no device code) |
| `test_loss_aggregation_generic.py` | 8 | Same math as above, device-parametrized (cpu + xpu on this machine) |
| `test_distributed_backend_generic.py` | 2 | Real distributed collective communication, generic across backends |
