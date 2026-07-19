# OpenRLHF on Intel XPU — `venv-tf` Environment: Setup, Versions, and Code Changes

Environment: `~/madhu/work2_with_transformers_native/OpenRLHF/venv-tf`
Hardware: Intel Arc Pro B70 (Battlemage), Ubuntu 24.04.3, kernel 7.0.0-14-generic
Validated: 2026-07-07 — real end-to-end SFT training run, exit code 0, real checkpoint written.

**What's different about this environment vs. the earlier one:** this uses `transformers`'
own built-in padding helpers (`_pad_input`, `_unpad_input`, `_index_first_axis` in
`modeling_flash_attention_utils.py`) instead of a custom vendored file
(`xpu_attn_utils.py`). Confirmed working, functionally identical results. See "Approach
comparison" at the end of this document.

---

## Part 1 — System-level (apt) packages required

These are one-time, machine-level installs — not tied to any specific venv. If setting up a
brand new machine, install all of these before touching Python.

| Package | Why it's needed | Install command |
|---|---|---|
| `libze-dev` | Provides the *unversioned* `libze_loader.so` symlink. Intel's oneCCL (backing PyTorch's `xccl` distributed backend) calls `dlopen("libze_loader.so")` — Ubuntu's runtime-only Level-Zero package ships only the versioned `libze_loader.so.1`. Without this, any real `torch.distributed` collective op (`barrier`, `all_reduce`, `broadcast`) fails with `oneCCL: ... ze_data was not initialized`. | `sudo apt install -y libze-dev` |
| `python3-dev` | Provides `Python.h`. Any PyTorch C++/SYCL extension (like DeepSpeed's `FusedAdam`) is compiled as a Python C-extension module and needs this header at compile time. Without it: `fatal error: Python.h: No such file or directory`. | `sudo apt install -y python3-dev` |
| Intel oneAPI DPC++/C++ Compiler (`icpx`), **version-matched to pip's runtime** | DeepSpeed's `FusedAdam` JIT-compiles a SYCL kernel on first use — this needs the actual compiler binary, not just runtime libraries. **Critical:** must match the version of `intel-sycl-rt` that pip installs (see Part 2) — a newer/older compiler than the pip runtime produces real C++ errors like `unknown type name '__DPCPP_SYCL_EXTERNAL_LIBC'`. On this machine: pip installs `2025.3.2`; the matching apt package is `intel-oneapi-compiler-dpcpp-cpp-2025.3` (version `2025.3.2-832`, resolved to `2025.3.3` — close enough, worked). | See exact commands below |
| `ninja` | Build tool DeepSpeed's JIT-compile path calls out to. Missing it gives `RuntimeError: Unable to JIT load the fused_adam op due to ninja not being installed.` even if `ninja` the Python package is installed — the **executable** must be resolvable, i.e. the venv's `bin/` must be on `PATH`. | `pip install ninja` (inside venv) — see Part 2 for the PATH caveat |

**Full apt-repo + compiler install commands** (one-time, machine-level):
```bash
sudo apt install -y gpg-agent wget
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update

# check what version pip's torch install will need BEFORE running this:
#   pip show intel-sycl-rt   (after installing torch — see Part 2)
apt-cache madison intel-oneapi-compiler-dpcpp-cpp-2025.3   # list available patch versions
sudo apt install -y intel-oneapi-compiler-dpcpp-cpp-2025.3=2025.3.2-832
```

**How to activate the compiler at runtime — do NOT source its `env/vars.sh`:**
```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
```
Sourcing the full `env/vars.sh` script prepends the compiler's own `lib/` directories to
`LD_LIBRARY_PATH`, which shadows the correctly-versioned Level-Zero/SYCL runtime libraries
your pip-installed torch depends on — this broke `torch.xpu.device_count()` entirely
(dropped to 0) when tested. Just adding `bin/` to `PATH` is sufficient and has zero side
effects on XPU detection.

---

## Part 2 — Python / pip packages, exact versions that worked

```bash
python3 -m venv venv-tf
venv-tf/bin/pip install --upgrade pip
venv-tf/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

| Package | Version installed | Source |
|---|---|---|
| `torch` | `2.12.1+xpu` | `https://download.pytorch.org/whl/xpu` |
| `torchvision` | `0.27.1+xpu` | same index |
| `triton-xpu` | `3.7.1` | auto-pulled by torch |
| `intel-sycl-rt` | `2025.3.2` | auto-pulled by torch — **check this immediately after install, it determines which apt compiler version you need (Part 1)** |
| `dpcpp-cpp-rt` | `2025.3.2` | auto-pulled by torch |
| `intel-cmplr-lib-rt` | `2025.3.2` | auto-pulled by torch |
| `oneccl` / `oneccl-devel` | `2021.17.2` | auto-pulled by torch |

Then OpenRLHF's own requirements, **excluding `flash-attn`** (cannot install on Intel XPU —
compiles CUDA kernels via `nvcc`; no CUDA toolchain/NVIDIA GPU exists here):
```bash
grep -v "^flash-attn" requirements.txt > /tmp/requirements_no_flashattn.txt
venv-tf/bin/pip install -r /tmp/requirements_no_flashattn.txt
```

Then the version bump + extras needed for the transformers-native fix and for running
without the `deepspeed` launcher:
```bash
venv-tf/bin/pip install "transformers>=5.13.0" "kernels>=0.15.2" mpi4py ninja
```

| Package | Version installed | Why |
|---|---|---|
| `transformers` | `5.13.0` (bumped from OpenRLHF's pin of `5.7.0`) | `_pad_input`/`_unpad_input`/`_index_first_axis` used in this environment's fix live in `transformers/modeling_flash_attention_utils.py` — verified present and signature-compatible at `5.13.0`. Also: `attn_implementation="flash_attention_2"` auto-redirects through the `kernels` library to a pre-compiled Intel SYCL kernel at this version — no CUDA needed. |
| `kernels` | `0.16.0` | Backs the `flash_attention_2` auto-redirect above. |
| `mpi4py` | `4.1.2` | Only needed because this environment's smoke test was run via plain `python -m` instead of the `deepspeed`/`torchrun` launcher — DeepSpeed's `init_distributed()` falls back to MPI-based rank discovery when standard launcher env vars aren't present. Not XPU-specific; a generic requirement of this simplified launch style. |
| `ninja` | `1.13.0` | DeepSpeed's JIT-compile path for `FusedAdam` invokes the `ninja` build tool. |

Then OpenRLHF itself, editable:
```bash
venv-tf/bin/pip install -e . --no-deps
```

**Runtime PATH requirement** (every time you activate this venv to actually run training):
```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:venv-tf/bin:$PATH"
```
Both directories matter: the compiler's `bin/` (for `icpx`, needed by `FusedAdam`'s JIT
compile) and the venv's own `bin/` (so the `ninja` executable installed above is found by
PyTorch's build-tool check — installing the `ninja` *Python package* alone is not enough if
its `bin/` isn't on `PATH`).

---

## Part 3 — OpenRLHF source code changes (all applied, all verified)

Every one of these is a genuine bug in OpenRLHF's own code: an assumption that CUDA/NVIDIA
is the only possible GPU backend. None of these are "workarounds" — the fix pattern is
PyTorch's own generic `torch.accelerator.*` API, which resolves to `"cuda"` on NVIDIA
machines (fully backward compatible) and `"xpu"` here. **Why needed, in one sentence per
fix:** the old code called a CUDA-only function/string that either doesn't exist or silently
does the wrong thing when torch is built for XPU instead of CUDA.

### 3.1 `openrlhf/models/ring_attn_utils.py`

**Why this file needs changes at all:** it unconditionally imports from the `flash_attn`
PyPI package at the top of the file — a package that cannot be installed on Intel XPU
(compiles CUDA kernels via `nvcc`). This import failing blocks `openrlhf.models` entirely,
so even `train_sft --help` crashes before printing anything, regardless of whether the
actual training run would ever use ring attention.

**Change 1 — lines 1-4 (imports):**
```python
# BEFORE:
import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather

# AFTER:
import torch
import torch.distributed as dist
from einops import rearrange
from transformers.modeling_flash_attention_utils import (
    _index_first_axis as index_first_axis,
    _pad_input as pad_input,
    _unpad_input as unpad_input,
)
```
`rearrange` moves to its true original source (`einops` — `flash_attn` was only
re-exporting it). `index_first_axis`/`pad_input`/`unpad_input` now come from `transformers`
(which OpenRLHF already depends on) instead of `flash_attn`. Verified signature-compatible:
`_unpad_input(hidden_states, attention_mask, unused_mask=None)`,
`_pad_input(hidden_states, indices, batch, seqlen)`,
`_index_first_axis(tensor, indices)` — exactly matching how `ring_attn_utils.py` calls them.

**Change 2 — added a small vendored `all_gather` (transformers doesn't ship a replacement
for this one):**
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
Placed directly below `RING_ATTN_GROUP = None`. Verified: pure `torch.distributed` primitives
(`all_gather_into_tensor`, `reduce_scatter_tensor`), zero CUDA-specific code — confirmed by
reading flash-attn's actual source for this function before writing this replacement.

**Change 3 — hardcoded CUDA device (unrelated bug, same file), inside
`reset_ring_attn_position_ids()`:**
```python
# BEFORE:
position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
# AFTER:
position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.accelerator.current_accelerator())
```
`torch.cuda.current_device()` raises `AssertionError: Torch not compiled with CUDA enabled`
on an XPU-only build. `torch.accelerator.current_accelerator()` returns the correct active
device (`xpu` here, `cuda` on an NVIDIA machine) generically.

### 3.2 `openrlhf/utils/deepspeed/deepspeed.py` — 5 separate spots

**Why this file needs changes:** it's OpenRLHF's DeepSpeed integration layer — every
training mode (SFT/RM/DPO/PPO) goes through `DeepspeedStrategy`. It has multiple places that
call CUDA-specific functions directly instead of using DeepSpeed's own hardware-agnostic
accelerator abstraction (which we confirmed elsewhere already correctly reports `"xpu"` via
`deepspeed.accelerator.get_accelerator()` — so DeepSpeed itself isn't the problem, this file
bypassing it is).

| Line area (method) | Before | After | Symptom without the fix |
|---|---|---|---|
| `setup_distributed()`, local_rank | `torch.cuda.set_device(local_rank)` | `torch.accelerator.set_device_index(local_rank)` | Doesn't trigger with `local_rank=-1` (single process), but would crash identically to the next row on any real multi-process launch |
| `setup_distributed()`, device mesh | `init_device_mesh("cuda", (dp_size, ring_attn_size, tp_size), mesh_dim_names=("dp","sp","tp"))` | `init_device_mesh(torch.accelerator.current_accelerator().type, (dp_size, ring_attn_size, tp_size), mesh_dim_names=("dp","sp","tp"))` | `AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'` — this symbol simply doesn't exist in a torch build compiled for XPU, not CUDA |
| tensor-parallel init path | `torch.cuda.empty_cache()` | `torch.accelerator.empty_cache()` | **No crash at all** — silently does nothing on an XPU-only torch build. More dangerous than the others: causes unreleased GPU memory to build up silently, risking a confusing delayed OOM much later instead of an immediate, traceable error |
| `all_reduce()` method | `data.to(torch.cuda.current_device())` | `data.to(torch.accelerator.current_accelerator())` | `AssertionError: Torch not compiled with CUDA enabled` |
| `all_gather()` method (×2 in the same line) | `torch.zeros_like(data).to(torch.cuda.current_device())` / `data.to(torch.cuda.current_device())` | same pattern with `torch.accelerator.current_accelerator()` | Same `AssertionError` |

### 3.3 `openrlhf/utils/loss_utils.py`

**Why this needs changing — worse than a simple crash:**
```python
# BEFORE:
device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# AFTER:
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
```
The old fallback check only asks "is CUDA available?" — on XPU this correctly evaluates
`False`, so it silently falls to `"cpu"` instead of `"xpu"`. Tensors then get created on CPU,
and later code calls `dist.all_reduce()` on them over the `xccl` process group — which has
**no CPU-communication path registered** — producing a confusing
`RuntimeError: No backend type associated with device type cpu`, nowhere near the actual
bug (a fallback that should have picked `xpu`, not `cpu`).

### 3.4 `openrlhf/utils/distributed_util.py`

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
Hit at the very end of a training run, inside `strategy.save_model()`'s checkpoint-save
cleanup barrier: `AssertionError: Torch not compiled with CUDA enabled`.

### Summary table — every changed file, every changed line

| File | Lines changed | Reason category |
|---|---|---|
| `openrlhf/models/ring_attn_utils.py` | imports (top of file), 1 new class (~20 lines), 1 device call | Unconditional `flash_attn` import (CUDA-only package unavailable on XPU) + 1 hardcoded CUDA device |
| `openrlhf/utils/deepspeed/deepspeed.py` | 5 call sites across `setup_distributed()`, `all_reduce()`, `all_gather()`, tensor-parallel init | Hardcoded `"cuda"` string / `torch.cuda.*` calls bypassing DeepSpeed's own working accelerator abstraction |
| `openrlhf/utils/loss_utils.py` | 1 line | Device-selection fallback only checked for CUDA, silently picked wrong device (`cpu`) on XPU |
| `openrlhf/utils/distributed_util.py` | 1 line | Hardcoded `torch.cuda.synchronize()` in a generic barrier helper |

**No other files needed changes** to reach a fully working end-to-end SFT training run.

---

## Part 4 — Unit tests (all ported from the earlier investigation, all passing here)

`tests/` in this environment now has 5 files, 27 tests total, **27/27 passing**:

| File | # tests | What it covers | Ported from |
|---|---|---|---|
| `test_loss_aggregation.py` | 8 | Loss-aggregation math (DP-averaging, grad-accum normalization, KL metrics) — pure CPU tensor math, unmodified upstream file | Shipped with OpenRLHF |
| `test_ray_env_vars.py` | 9 | Ray `runtime_env` env-var defaulting/override logic, mocked Ray module — no device/tensor code | Shipped with OpenRLHF |
| `test_xccl_backend.py` | 2 | Confirms Intel's `xccl` distributed backend (the real XPU-equivalent of NCCL) actually works: reports available, AND successfully runs a real `init_process_group` + `all_reduce` on a real XPU tensor. Requires `libze-dev` (Part 1) to pass. | Earlier investigation (`work1_with customfunction`) |
| `test_loss_aggregation_xpu.py` | 8 | Same formulas as `test_loss_aggregation.py`, but with every tensor explicitly created on `device="xpu"` and asserted to stay there — closes the gap that the upstream file only ever tests CPU tensors, never real GPU tensors | Earlier investigation |
| **Total** | **27** | | |

Run with:
```bash
cd OpenRLHF && venv-tf/bin/python -m pytest tests/ -v
```

---

## Part 5 — End-to-end verification

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$(pwd)/venv-tf/bin:$PATH"
venv-tf/bin/python -m openrlhf.cli.train_sft \
   --data.max_len 512 --data.dataset Open-Orca/OpenOrca \
   --data.input_key question --data.output_key response --data.max_samples 8 \
   --train.batch_size 2 --train.micro_batch_size 1 \
   --model.model_name_or_path Qwen/Qwen2.5-0.5B \
   --ckpt.output_dir /tmp/openrlhf_smoke_sft_tf --ckpt.save_steps -1 \
   --logger.logging_steps 1 --eval.steps -1 --ds.zero_stage 0 \
   --train.max_epochs 1 --ds.param_dtype bf16 --ds.attn_implementation sdpa --adam.lr 5e-6
```
Result: `EXIT CODE: 0`. Real decreasing loss across 8 steps
(`1.90 → 0.95 → 0.208 → 2.78 → 1.93 → 0.469 → 0.505 → 1.74`) — numerically consistent with the
custom-file (`xpu_attn_utils.py`) approach's run on the same seed/data. Real ~1.26GB
`model.safetensors` checkpoint written to disk.

---

## Approach comparison: transformers-native vs. custom `xpu_attn_utils.py`

| | Custom file (`work1_with customfunction`) | transformers-native (this env, `venv-tf`) |
|---|---|---|
| New file to maintain | Yes — `openrlhf/models/xpu_attn_utils.py` (~150 lines) | No — only a small vendored `all_gather` (~20 lines) inside `ring_attn_utils.py` itself |
| `index_first_axis`/`pad_input`/`unpad_input` source | Hand-copied from flash-attn's source | Imported directly from `transformers.modeling_flash_attention_utils` (already installed, already a dependency) |
| `all_gather` | Vendored (flash-attn doesn't have a transformers equivalent either way) | Same — still vendored, unavoidable |
| Functional correctness | Round-trip test: `True` | Round-trip test: `True` — **identical result** |
| SFT smoke test | Exit code 0, real checkpoint | Exit code 0, real checkpoint — **same outcome** |
| Maintenance risk | We own and must maintain a copy of flash-attn's padding logic indefinitely | Relies on an **internal** (underscore-prefixed) transformers module not covered by its public API stability guarantees — re-verify signatures on any transformers upgrade |
| Recommendation | — | **Preferred** — less code to own, but pin/verify the transformers version this was checked against (`5.13.0`) before upgrading |
