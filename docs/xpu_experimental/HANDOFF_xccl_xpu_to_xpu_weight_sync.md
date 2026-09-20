# HANDOFF: XCCL direct XPU→XPU weight-sync (finish on a 2-XPU machine)

**Audience:** an engineer or coding agent continuing this on a machine with **≥2 physical Intel
XPUs**. Everything below is fact-based from work done on a single-XPU box; the single-XPU box
**cannot** finish this (XCCL needs distinct devices per rank).

---

## Goal

Weight-sync = after each training step, broadcast updated model weights from the trainer
(rank 0) to the vLLM engine(s). Add a **direct XPU→XPU** path using Intel **XCCL** (oneCCL) so
weights move device-to-device with **no CPU staging** (faster than the current gloo path).

Backends, by platform:
- CUDA/ROCm → `nccl` (vLLM's `PyNcclCommunicator`) — unchanged.
- Intel XPU, **multi-device** → `xccl` (this work) — direct, fast.
- Intel XPU **single-device** or no-XCCL → `gloo` (CPU-staged) — the safe fallback, already shipped.

---

## Current state (already implemented in the repo)

All in **`openrlhf/utils/distributed_util.py`** (read it — the code is present and syntactically valid):

1. **`_XcclBroadcastCommunicator`** (class) — `.broadcast(tensor, src, stream=None)` does a real
   XCCL broadcast on the XPU tensor in place, returns the same tensor. Same interface as the
   gloo / PyNccl communicators.
2. **`_init_xccl_process_group(master_address, master_port, rank, world_size)`** — builds a
   stateless XCCL `ProcessGroup` by mirroring vLLM's gloo/NCCL pattern
   (`ProcessGroup` + `ProcessGroupXCCL` + `_register_backend(torch.device("xpu"), XCCL, ...)`).
   This is needed because **vLLM's XPU platform does NOT implement
   `stateless_init_device_torch_dist_pg`** (it raises `NotImplementedError`), so we cannot use
   vLLM's helper and build the group ourselves.
3. **`resolve_vllm_sync_backend(backend)`** — accepts `"nccl" | "gloo" | "xccl" | None`.
   - `None` (auto) → `nccl` on CUDA/ROCm, else **`gloo`** (NOT xccl — see "Why auto stays gloo").
   - `"xccl"` is honored only on an XPU build with XCCL available, else raises.
4. **`stateless_init_process_group(..., backend=None)`** — has an `if backend == "xccl":` branch
   that returns `_XcclBroadcastCommunicator`.
5. Config flag **`--vllm.sync_backend`** in `openrlhf/cli/train_ppo_ray.py` accepts
   `choices=("nccl","gloo","xccl")`, `default=None`.

**Callers already thread the backend through:**
- `openrlhf/trainer/ray/ppo_actor.py`: resolves once (`self.vllm_sync_backend`), passes to
  `stateless_init_process_group(..., backend=...)`, and the broadcast picks
  `stream = current_stream() if nccl else None` (xccl gets None → ignored, correct).
- `openrlhf/trainer/ray/vllm_worker_wrap.py`: stores `self._sync_backend`, passes `backend=` to
  `stateless_init_process_group`, same stream logic.

**Verified on single XPU:** an *isolated* 2-rank XCCL broadcast of an XPU tensor returns the
correct value in place, no CPU staging (both ranks shared device 0). This proved the code path
works in isolation.

---

## THE PROBLEM to fix (why it's not done)

On a **single physical XPU**, a **full GRPO training run** with `--vllm.sync_backend xccl`
**HANGS** (0 training steps, times out). Root cause, from the runtime warning:

```
ProcessGroupXCCL.cpp:1916 ... using GPU 0 as device used by this process is currently unknown.
This can potentially cause a hang if this rank to GPU mapping is incorrect.
You can specify device_id in init_process_group() to force use of a particular device.
```

Two things are happening:
1. **Single-XPU is degenerate for XCCL** — trainer (rank 0) and vLLM (rank 1) both map to
   device 0. XCCL is a *device-to-device* collective; two ranks on one physical chip
   deadlock in a real (concurrent) training loop. gloo avoids this by routing through CPU.
2. **The group is built WITHOUT a `device_id`** — `_init_xccl_process_group` does not tell the
   process group which XPU each rank owns, so XCCL can't establish the rank→device mapping.

**On ≥2 XPUs this is expected to work** (each rank has its own device, which is what XCCL wants),
**but only if we pass the per-rank `device_id`** so the mapping is explicit.

---

## UPDATE (fix already applied in code — needs 2-XPU validation)

The `device_id` fix and the single-XPU safety fallback described below have now been
**implemented** in `distributed_util.py`:
- `_init_xccl_process_group(..., device)` now takes the rank's device, calls
  `torch.xpu.set_device(index)`, sets `pg.bound_device_id`, and registers the backend for the
  specific `xpu:index` (not generic `"xpu"`). This targets the "device unknown → hang" warning.
- `resolve_vllm_sync_backend` now **falls back to gloo (with a warning) when `xccl` is requested
  but `torch.xpu.device_count() < 2`**, so single-XPU never hangs. XCCL only truly engages on
  ≥2 physical XPUs.
- Unit tests updated (device-free resolver tests pass). What remains is **real 2-XPU validation**
  (the isolated broadcast test + E2E GRPO with `--vllm.sync_backend xccl`), which cannot run on a
  single-XPU box.

The original analysis and remaining steps are preserved below for context.

## THE FIX TO TRY (do this on the 2-XPU box)

### Change 1 — pass `device_id` into `_init_xccl_process_group`
File: `openrlhf/utils/distributed_util.py`, function `_init_xccl_process_group` (~line 65).

Add a `device` parameter and bind it. The `ProcessGroupXCCL` / `_register_backend` must reference
the **rank's own XPU device index**, not a bare `torch.device("xpu")`. Concretely:

```python
def _init_xccl_process_group(master_address, master_port, rank, world_size, device):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, ProcessGroupXCCL

    # device is the rank's OWN XPU, e.g. torch.device("xpu", local_rank). Set it BEFORE building
    # the group so XCCL knows the rank->device mapping (fixes the "device unknown -> hang" warning).
    torch.xpu.set_device(device.index if device.index is not None else 0)

    timeout = timedelta(seconds=1800)
    store = dist.TCPStore(master_address, master_port, world_size, is_master=(rank == 0), timeout=timeout)
    prefix_store = PrefixStore("openrlhf-xccl", store)

    pg = ProcessGroup(prefix_store, rank, world_size)
    backend_class = ProcessGroupXCCL(prefix_store, rank, world_size)   # may accept a device/options arg — check the torch build
    backend_type = ProcessGroup.BackendType.XCCL
    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()
    pg._register_backend(device, backend_type, backend_class)          # register for the SPECIFIC xpu:index, not generic "xpu"
    return pg
```

### Change 2 — pass the real device from the caller
File: `openrlhf/utils/distributed_util.py`, `stateless_init_process_group`, the `xccl` branch (~line 166):
```python
    if backend == "xccl":
        pg = _init_xccl_process_group(master_address, master_port, rank, world_size, device)  # forward device
        logger.info("vLLM weight-sync backend: xccl (direct XPU-to-XPU)")
        return _XcclBroadcastCommunicator(pg, device)
```
`device` already arrives here (ppo_actor builds `torch.device(accelerator.type, current_device_index())`
and vllm_worker_wrap passes `self.device`). Confirm each rank's `device` is its **own** XPU
(index differs per rank) on the 2-XPU box — that is the whole point.

### Change 3 (investigate) — `ProcessGroupXCCL` constructor options
Different torch builds expose `ProcessGroupXCCL(store, rank, size)` OR require an options/device
argument. On the 2-XPU box, check:
```python
python -c "from torch.distributed.distributed_c10d import ProcessGroupXCCL; help(ProcessGroupXCCL.__init__)"
```
If it accepts a device or `Options` with a timeout, pass them (mirror how `cuda.py`'s
`stateless_init_device_torch_dist_pg` passes `ProcessGroupNCCL.Options()`).

---

## Single-XPU fallback (MUST remain intact)

The safe default is already in place and must not regress:
- `resolve_vllm_sync_backend(None)` returns **`gloo`** on XPU (NOT xccl). So a single-XPU user who
  runs without `--vllm.sync_backend` keeps the working gloo path. **Do not change auto to xccl.**
- `xccl` is **opt-in only** (`--vllm.sync_backend xccl`).
- If you want, add a guard: when `xccl` is requested but `torch.xpu.device_count() < 2`, either
  log a loud warning or fall back to gloo — because xccl on a single device hangs. (Currently the
  resolver allows explicit xccl on single XPU; consider blocking it.)
- **verl-style alternative** (if xccl proves unworkable): a colocated ZMQ + CPU-shared-memory
  transfer is the other CPU-staged option (see kahlun/verl PR #3), but it is the same speed class
  as gloo — not worth adopting unless gloo is removed.

---

## How to VALIDATE on the 2-XPU box (acceptance criteria)

1. **Isolated collective** (fast sanity):
   run `tests/test_gloo_weight_sync.py::test_xccl_broadcast_direct_on_xpu` — it auto-skips unless
   `torch.xpu.device_count() >= 2`. On 2 XPUs it must PASS: rank 1 receives rank 0's value, tensor
   stays on `xpu:*` (no CPU), in place.
2. **Full unit suite:** `pytest tests/ -q` — all pass, no hang.
3. **E2E GRPO with xccl:**
   ```
   ray start --head --num-gpus=2
   python -m openrlhf.cli.train_ppo_ray ... --vllm.sync_backend xccl --vllm.num_engines 1
   ```
   Must reach ≥2 Global steps, log `vLLM weight-sync backend: xccl (direct XPU-to-XPU)`, show
   `timing/broadcast` each step, and **weights must actually change** in vLLM (rewards move; or
   instrument: record a param, sync, re-read on the engine, assert changed).
4. **Compare speed vs gloo:** run the same config with `--vllm.sync_backend gloo` and compare
   `timing/broadcast` — xccl should be faster (this is the whole point; currently unmeasured).
5. **Confirm no NVIDIA/single-XPU regression:** default (no flag) still auto-selects gloo on XPU /
   nccl on CUDA.

---

## Key files (exact locations to change / review)

| File | What's there | Change needed |
|---|---|---|
| `openrlhf/utils/distributed_util.py` | `_init_xccl_process_group` (~L65), `_XcclBroadcastCommunicator` (~L42), resolver (~L90), `stateless_init_process_group` xccl branch (~L162) | **Change 1 + 2** (device_id) |
| `openrlhf/trainer/ray/ppo_actor.py` | resolves backend (~L110), builds `device` + calls `stateless_init_process_group` (~L160), broadcast stream (~L451) | verify each rank's device is its own XPU on multi-XPU |
| `openrlhf/trainer/ray/vllm_worker_wrap.py` | `self._sync_backend` (L14), `stateless_init_process_group(..., backend=)` (L21), buffer `device=self.device` (L43) | should work as-is; verify device per rank |
| `openrlhf/cli/train_ppo_ray.py` | `--vllm.sync_backend` flag `choices=(nccl,gloo,xccl)`, `default=None` (~L245) | none |
| `tests/test_gloo_weight_sync.py` | `test_xccl_broadcast_direct_on_xpu` (skips if <2 XPU), resolver tests | run on 2-XPU |

---

## Environment notes (important)
- On a single XPU, running xccl repeatedly can **wedge the XPU driver** (device stops responding;
  even `torch.ones(device='xpu')` hangs) — recovery required a **machine reboot**. On the 2-XPU
  box this may be less likely, but if the device wedges, reboot before continuing.
- oneAPI on PATH: `export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"`, run with
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1` (both devices visible).
- CCL env knobs seen in runs: `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` (topology warning),
  `CCL_ATL_TRANSPORT=ofi`. May be needed for stable multi-XPU XCCL.

---

## TL;DR for the agent
1. Read `openrlhf/utils/distributed_util.py` — the xccl code is implemented but builds the group
   **without a device_id**, which hangs on single-XPU and needs the per-rank device to work on
   multi-XPU.
2. Apply **Change 1 + 2** (thread `device` into `_init_xccl_process_group`, register the backend
   for the rank's specific `xpu:index`, `set_device` before building).
3. Keep auto-select = gloo on single XPU (do not regress); xccl stays opt-in.
4. Validate with `test_xccl_broadcast_direct_on_xpu` + E2E GRPO `--vllm.sync_backend xccl` on
   **2 XPUs**, and measure speed vs gloo.
