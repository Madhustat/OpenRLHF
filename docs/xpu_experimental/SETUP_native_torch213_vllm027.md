# Setup: Native torch 2.13 + vLLM 0.27 (Intel XPU, no Docker)

Reproduction guide for the environment validated 2026-09-02 through
2026-09-06: native (non-container) torch 2.13.0+xpu + vLLM 0.27.2.dev0
(built from the `v0.27.1` tag), on 2× Intel Arc Pro B70. This is the
environment `resolve_vllm_sync_backend()`'s xccl auto-detection (branch
`xpu/xccl-auto-detect`) was validated against — see `XCCL_KNOWN_ISSUES.md`
Bug 1 update and this repo's `tests/` suite.

This supersedes `SETUP_new_machine_from_scratch.md` (torch 2.12.0+xpu) for
any work that wants XCCL — torch 2.12 segfaults in `ProcessGroupXCCL`
(torch-xpu-ops#4238); 2.13 does not. Keep both environments if you need to
compare — see Part 0.

---

## Part 0 — Do not touch an existing torch-2.12 environment

If you already have a working torch-2.12 venv (e.g. `venv-tf` from
`SETUP_new_machine_from_scratch.md`), build this as a **separate, new venv**.
Do not upgrade torch in place, `cp -r` the old venv, or symlink packages
between them — the two stacks pull in different `intel-sycl-rt` versions
(`libsycl.so.8` for 2.12, `libsycl.so.9` for 2.13) and mixing them produces
`undefined symbol` errors that look unrelated to the actual cause.

```bash
python3 -m venv ~/venvs/openrlhf-native-torch213-xpu
```

---

## Part 1 — Install torch 2.13 first, check `LD_LIBRARY_PATH`

```bash
unset LD_LIBRARY_PATH   # do this before every command below, every new shell
~/venvs/openrlhf-native-torch213-xpu/bin/pip install --upgrade pip
~/venvs/openrlhf-native-torch213-xpu/bin/pip install torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/xpu
```

**Gotcha:** a stale `LD_LIBRARY_PATH` inherited from an earlier shell session
(e.g. from exploring the torch-2.12 install, or from sourcing the oneAPI
compiler's `env/vars.sh`) will make the fresh venv load the *system's* older
Unified Runtime loader instead of its own bundled one, producing
`undefined symbol: urDeviceWaitExp` on `import torch`. `unset
LD_LIBRARY_PATH` before every command in this venv fixes it — there is no
persistent fix short of that, since the contamination comes from your shell
environment, not the venv itself.

**Verify:**
```bash
unset LD_LIBRARY_PATH
~/venvs/openrlhf-native-torch213-xpu/bin/python -c "
import torch
print(torch.__version__)              # expect 2.13.0+xpu
print(torch.xpu.device_count())        # expect 2 (or however many physical XPUs)
"
```

---

## Part 2 — oneAPI compiler on `PATH` (for DeepSpeed's JIT-built ops)

DeepSpeed's `FusedAdam`/`CPUAdam` JIT-compile via `icpx`. Check what's
already installed before assuming you need to install anything:

```bash
ls /opt/intel/oneapi/compiler/*/bin/icpx 2>/dev/null
```

If present (this box already had `2025.3` and `2026.0`), just add it to
`PATH`:
```bash
export PATH="/opt/intel/oneapi/compiler/2026.0/bin:$PATH"
```
If not present, follow `SETUP_new_machine_from_scratch.md` Part 2/4 to add
Intel's apt repo and install the compiler matching your `intel-sycl-rt`
version.

**Known caveat (see `XCCL_KNOWN_ISSUES.md` and this doc's Part 6):**
DeepSpeed's icpx-JIT-built `CPUAdam` extension **segfaults at `.so` load
time** on this stack for reasons unrelated to torch 2.13 vs 2.12 (an
icpx/torch-2.13 ABI issue, not yet root-caused). Part 6 below covers the
workaround.

---

## Part 3 — Build vLLM for XPU, from source, against torch 2.13

Clone as a **sibling** of your OpenRLHF checkout, not nested inside it —
nesting breaks `import vllm`'s namespace-package resolution (same gotcha as
`SETUP_new_machine_from_scratch.md` Part 8).

```bash
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
git checkout v0.27.1
```

vLLM's `requirements/xpu.txt` pins `torch==2.13.0` directly — good, it
matches what you just installed, and the sdist build would otherwise try to
pull a fresh plain-PyPI torch regardless of `VLLM_TARGET_DEVICE`. Strip the
torch/torchvision/torchaudio pins so the build uses your already-installed
torch instead of re-resolving:

```bash
python use_existing_torch.py   # vLLM's own helper script; edits requirements/*.txt + pyproject.toml in place
export VLLM_TARGET_DEVICE=xpu
~/venvs/openrlhf-native-torch213-xpu/bin/pip install -v -r requirements/xpu.txt
~/venvs/openrlhf-native-torch213-xpu/bin/pip install --no-build-isolation --no-deps -e .
```

`requirements/xpu.txt` also pulls Intel's prebuilt `vllm_xpu_kernels` wheel —
no local kernel compilation needed.

### Gotcha: a bare `pip install vllm==0.27.1` (no index) installs the wrong build
If you (or a script) ever runs plain `pip install vllm==<version>` without
`VLLM_TARGET_DEVICE=xpu` and the source build above, pip resolves the
generic CUDA-targeted wheel — it pulls in ~40 NVIDIA/CUDA packages
(`flashinfer-python`, `nvidia-cuda-*`, `nvidia-cutlass-dsl-*`,
`torchvision`/`torchaudio`/`torchcodec` CUDA builds, plain `triton`) and
silently replaces nothing in torch itself, so `torch.__version__` still
looks right — the wrongness only shows up later as import-time crashes.
**Symptoms and fixes if this happens:**
- `AssertionError: Torch not compiled with CUDA enabled` inside
  `flashinfer/gdn_kernels/...` at import time → uninstall `flashinfer-python`
  and the CUDA-only leftovers (`cuda-bindings`, `cuda-core`, `cuda-pathfinder`,
  `cuda-python`, `cuda-tile`, `nvidia-cuda-*`, `nvidia-cutlass-dsl-*`,
  `nvidia-nvvm`, `tilelang`).
- `cannot import name 'intel' from 'triton._C.libtriton'` → the plain
  `triton` wheel overwrote `triton-xpu`'s `libtriton.so` in the shared
  `site-packages/triton/` path (both packages install to the same location).
  Fix: `pip uninstall -y triton && pip install --force-reinstall --no-deps
  triton-xpu==3.7.2 --index-url https://download.pytorch.org/whl/xpu`.

**Verify the build:**
```bash
unset LD_LIBRARY_PATH
~/venvs/openrlhf-native-torch213-xpu/bin/python -c "import vllm; print(vllm.__version__)"
# expect something like 0.27.2.dev0+g<sha>.d<date>.xpu
```

---

## Part 4 — OpenRLHF dependencies

Same as `SETUP_new_machine_from_scratch.md` Part 6 — install
`requirements.txt` minus `flash-attn` (no CUDA toolchain on XPU), plus
`transformers`/`kernels`/`mpi4py`/`ninja`/`pytest` as needed by whichever
OpenRLHF checkout/branch you're testing. `deepspeed==0.19.5` and
`transformers` in the 5.15-5.16.x range are what this environment carries;
exact pins depend on which branch's `requirements.txt` you install from.

```bash
unset LD_LIBRARY_PATH
~/venvs/openrlhf-native-torch213-xpu/bin/pip install -e <your-openrlhf-checkout> --no-deps
~/venvs/openrlhf-native-torch213-xpu/bin/pip install pytest pytest-asyncio
```

---

## Part 5 — Verify torch + XPU + Ray env vars

Same two-variable pattern as the torch-2.12 setup:
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1     # both XPUs visible; :0 alone for single-GPU testing
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1  # Ray misdetects Arc Pro B70 as NVIDIA otherwise
```

```bash
unset LD_LIBRARY_PATH
~/venvs/openrlhf-native-torch213-xpu/bin/python -c "
import torch
print('torch:', torch.__version__)
print('xpu.is_available():', torch.xpu.is_available())
print('xpu.device_count():', torch.xpu.device_count())
"
```
Expected: `2.13.0+xpu`, `True`, `2`.

---

## Part 6 — DeepSpeed CPUAdam segfault workaround (icpx/torch-2.13, unresolved root cause)

DeepSpeed's default path (`--ds.adam_offload`) JIT-builds `CPUAdam` via
`icpx`. On this stack, loading the freshly-built `.so` segfaults at
`torch.utils.cpp_extension._import_module_from_library`. This is **not**
torch-2.12-vs-2.13 related — it's a separate, unresolved icpx/torch-2.13 ABI
issue. Workaround, no code changes needed — this env var is already read by
`openrlhf/utils/deepspeed/deepspeed.py`:

```bash
export OPENRLHF_DS_TORCH_ADAM=1   # forces plain torch.optim.AdamW instead of DeepSpeed's CPUAdam
```

---

## Part 7 — Known-good launch pattern

Every real training run in this validation used this combination:
```bash
unset LD_LIBRARY_PATH
export PATH="/opt/intel/oneapi/compiler/2026.0/bin:~/venvs/openrlhf-native-torch213-xpu/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1
ray start --head --num-gpus=2
python -m openrlhf.cli.train_ppo_ray ...
```

For colocated (actor + vLLM sharing GPUs via sleep/wake) runs, add
`--train.colocate_all --vllm.enable_sleep --ds.enable_sleep`. Weight-sync
backend: omit `--vllm.sync_backend` to auto-detect (xccl on >=2 physical
XPUs with this torch version, per `xpu/xccl-auto-detect`), or force it
explicitly with `--vllm.sync_backend gloo|xccl`. See `XCCL_KNOWN_ISSUES.md`
Bug 7 for one specific combination (ZeRO-3 + colocate_all + xccl + critic)
where gloo should still be forced explicitly.

## Part 8 — Cleanup pattern between runs

Ray sometimes leaves orphaned `VLLM::EngineCore` processes holding GPU
memory after a crashed run, which `ray stop` alone doesn't reap. Full clean
sequence between runs:
```bash
ray stop --force
rm -rf /tmp/ray
for pid in $(pgrep -f "VLLM::EngineCore"); do kill -9 "$pid"; done
xpu-smi ps            # expect "No data"
xpu-smi discovery | grep -i state   # expect "Device State: normal" on all devices
```
