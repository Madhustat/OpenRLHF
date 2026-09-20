# OpenRLHF on Intel XPU — Install Guide (verified torch 2.13 stack)

This guide reproduces the **exact environment that was validated end-to-end**
on Intel Arc Pro B70. Every version below is read from the working validation
environment, not inferred and not a PyPI guess.

> **Do not treat vLLM as a PyPI release.** The validated vLLM is an editable
> source build from a specific commit. There is no `0.27.2` XPU wheel on PyPI —
> that string is a source-build self-version. See Step 5.

> **Scope of verification.** The version table below is read directly from the
> live venv that produced every passing result on this branch (`torch 2.13`).
> The step-by-step procedure that follows was carried over from the earlier
> `torch 2.12` revision of this guide and adjusted for these versions; the
> individual steps have not been re-executed from scratch on a clean machine
> since the 2.13 upgrade. Trust the table exactly; treat the steps as a close
> guide rather than a byte-verified script.

---

## Verified stack (exact, from the torch 2.13 validation venv)

| Component | Exact version | Source |
|---|---|---|
| Hardware | 2× Intel Arc Pro B70 (Battlemage, PCIe, no Xe Link) | — |
| Kernel | `7.0.0-31-generic` (`xe` driver) | — |
| Compute runtime | `intel-opencl-icd`, `libze-intel-gpu1` == `26.22.38646.6-1~24.04~ppa1` | Intel PPA |
| Level Zero loader | `libze1 == 1.28.6-1~24.04~ppa1`, `level-zero 1.27.0` | Intel PPA |
| Python | 3.13.12 | miniforge base |
| PyTorch | `torch==2.13.0+xpu` | pytorch.org XPU index |
| torchvision / torchaudio | **not installed** — not required by this stack | — |
| triton | `triton-xpu==3.7.2` | pytorch.org XPU index |
| vLLM | `0.27.2.dev0+g6e448d0ea` (commit `6e448d0`) | **editable source build** (Step 5) |
| vllm-xpu-kernels | `0.1.12` wheel | GitHub release (Step 5) |
| Ray | `ray==2.55.0` | PyPI |
| DeepSpeed | `deepspeed==0.19.1` | PyPI |
| transformers | `5.7.0` | PyPI |
| oneCCL | `oneccl==2022.0.0`, `oneccl-devel==2022.0.0` | PyPI |
| Intel runtime | `intel-sycl-rt / -cmplr-lib-rt / -cmplr-lib-ur / -cmplr-lic-rt / -opencl-rt / -openmp == 2026.0.0` | oneAPI 2026.0 |
| oneMKL | `mkl`, `onemkl-sycl-{blas,dft,lapack,rng,sparse} == 2026.0.0` | PyPI |
| Intel MPI | `impi-rt==2021.18.0` | PyPI |
| intel-pti | `0.17.0` | PyPI |
| numpy | `2.4.6` | PyPI |

Not used on XPU (see notes): **flash-attn is NOT installed** (CUDA-only).
`bitsandbytes==0.50.2` is present but unused — do not let it pull in a CUDA torch.

### Two deliberate absences — do not "fix" these

| Absent | Why, and what it costs you |
|---|---|
| `icpx` (Intel compiler) | Not installed. DeepSpeed therefore cannot JIT-build `FusedAdam`, so OpenRLHF must run with `OPENRLHF_DS_TORCH_ADAM=1` (plain `torch.optim.AdamW`). See [ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md) — `--ds.enable_sleep` used to hard-crash because of this and is now handled with a capability fallback. |
| `dpctl` | Deliberately **not** installed: it pulls Intel 2026.x runtime libs that conflict with this stack. Consequence: Ray's `IntelGPUAcceleratorManager` cannot discover XPUs and reports `GPU=0`, so **every Ray placement group hangs** unless you start the head explicitly with `ray start --head --num-gpus 2`. |

---

## Step 0 — Verify hardware and driver

```bash
lspci | grep -i display        # should list the Arc Pro B70 device(s)
sudo xpu-smi discovery         # or: clinfo | grep "Device Name"
```

---

## Step 1 — Base: use Intel's XPU PyTorch container

The validated box ran inside Intel's XPU image with the GPU devices mapped in.
This is strongly preferred over a bare-metal driver install.

```bash
docker run -it --rm \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  --shm-size=16g \
  -v $HOME/OpenRLHF:/workspace/OpenRLHF \
  intel/intel-extension-for-pytorch:xpu \
  /bin/bash
```

`--device /dev/dri` and the `render`/`video` groups are mandatory — without
them `torch.xpu.is_available()` returns False.

> If your base image ships an older torch, you MUST replace it with **2.13.0+xpu**
> in Step 4. The validated stack is **2.13**.

---

## Step 2 — oneAPI runtime

The Intel runtime pip packages (`intel-sycl-rt==2026.0.0` etc.) are pulled in
automatically by the torch-xpu wheels in Step 4 — **do not install them separately.**

There is **no oneAPI compiler on the validated box** — `icpx` is absent, and this is
intentional (see "Two deliberate absences" above). If you do put a compiler on PATH,
DeepSpeed will try to JIT-build `FusedAdam`; that path has not been validated here.

### Loader-path ordering (load-bearing)

The venv's `lib/` must come **before** any base conda/miniforge `lib/` on
`LD_LIBRARY_PATH`. A base miniforge3 install ships an older `libur_loader`, which makes
torch 2.13's `libsycl.so.9` fail at import with:

```
undefined symbol: urDeviceWaitExp
```

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
```

---

## Step 3 — Python environment (3.13, matching validated)

```bash
conda create -n openrlhf-xpu python=3.13 -y
conda activate openrlhf-xpu
```

> The validated env is Python **3.13.12**. If your base image pins 3.12, that is
> usually fine, but 3.13 is the exercised version.

---

## Step 4 — Install PyTorch 2.13.0+xpu

`torchvision` and `torchaudio` are **not installed** in the validated env and are not
required — do not add them just to mirror a CUDA setup.

```bash
pip install torch==2.13.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu

pip install triton-xpu==3.7.2
```

Verify XPU is live and torch is the XPU build:

```bash
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available(), 'count:', torch.xpu.device_count())"
# expect: 2.13.0+xpu  /  xpu: True  count: 2
```

If `xpu: False`: re-check `/dev/dri` mapping, render/video groups, and PATH.

---

## Step 5 — Build vLLM for XPU from the exact commit

There is **no wheel** — the validated vLLM was built editable from source at a
pinned commit, with `VLLM_TARGET_DEVICE=xpu`, against the torch 2.13 already
installed in Step 4.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 6e448d0ea      # == v0.27.2.dev0+g6e448d0ea

# --no-build-isolation forces the build to use the XPU torch already in the venv
# instead of pulling a fresh CUDA torch into an isolated build environment.
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation
```

Install the matching XPU kernels wheel:

```bash
pip install "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.12/vllm_xpu_kernels-0.1.12-cp38-abi3-manylinux_2_28_x86_64.whl"
```

If `_xpu_C` fails to `dlopen` a kernel `.so` at runtime, add the
`vllm_xpu_kernels` install directory to `LD_LIBRARY_PATH`.

Verify vLLM did not disturb torch:

```bash
python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)"
# expect: torch 2.13.0+xpu  /  vllm 0.27.2.dev0+g6e448d0ea...
```

If torch now reads `+cuNNN`, vLLM pulled a CUDA build — reinstall Step 4 and
rebuild vLLM with `--no-build-isolation`.

---

## Step 6 — Install OpenRLHF (this fork, XPU branch)

```bash
git clone https://github.com/Madhustat/OpenRLHF.git
cd OpenRLHF
git checkout fix/ppo-zero3-deepspeed-sleep

# --no-deps so the install does NOT pull a CUDA torch or a different vLLM
# over the XPU builds from Steps 4-5.
pip install -e . --no-deps

# then install the remaining pure-Python deps
pip install "ray==2.55.0" deepspeed==0.19.1 transformers==5.7.0
```

> `fix/ppo-zero3-deepspeed-sleep` is `openrlfh_exp_multi` plus the ZeRO-3 /
> DeepSpeed-sleep fixes described in
> [ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md). Use
> `openrlfh_exp_multi` instead if you specifically want the pre-fix baseline.

Dependency notes for XPU:

- **flash-attn: do NOT install.** It is CUDA-only. OpenRLHF runs without it on
  XPU (the validated env has no flash-attn).
- **bitsandbytes: skip / leave unused.** It is CUDA-oriented and not exercised by
  any XPU path here. If installed, ensure it does not drag in a CUDA torch.

---

## Step 7 — Runtime environment variables

Weight sync on XPU uses **gloo** by default — no CCL env vars required for the
production path. The variables below are what the E2E suite exports.

```bash
export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:$LD_LIBRARY_PATH"   # venv lib FIRST (see Step 2)
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1          # both XPUs visible
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 # else actor + vLLM collide on xpu:0
export OPENRLHF_DS_TORCH_ADAM=1                        # no icpx -> no FusedAdam JIT build
export OPENRLHF_WEIGHT_PROBE=0
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
```

`RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` is **mandatory** on XPU —
without it Ray overrides the device selector and the DeepSpeed actor and the
vLLM engine both land on `xpu:0`.

`OPENRLHF_DS_TORCH_ADAM=1` is required on this box because `icpx` is absent, so
DeepSpeed cannot JIT-build `FusedAdam`. With ZeRO-3 + `--ds.enable_sleep` this used
to crash on a `FusedAdam`-only assertion; see
[ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md).

Ray cannot auto-discover Intel XPUs without `dpctl` (deliberately not installed), so
**start the head with the GPU count declared explicitly** or every placement group
hangs on `{'GPU': 1.0}`:

```bash
ray start --head --num-gpus 2
```

The XCCL-only env vars (`CCL_ZE_IPC_EXCHANGE`, `FI_PROVIDER=shm`, etc.) are
**not needed** for the gloo path. See [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md).

---

## Step 8 — Datasets and models the suite expects

| Used as | Path in suite | How provided |
|---|---|---|
| RL prompts | `tests/data/gsm8k_train_prompts.jsonl` | **auto-generated** by `tests/prepare_e2e_data.py` from GSM8K on first run |
| SFT data | `tests/data/gsm8k_sft/train.parquet` | **auto-generated** by the same script from GSM8K on first run |
| RM/DPO data | `OpenRLHF/preference_dataset_mixture2_and_safe_pku` | HF hub, auto-downloaded |
| VLM data | `hiyouga/geometry3k` | HF hub, auto-downloaded |
| RL/SFT/RM/DPO model | `Qwen/Qwen2.5-0.5B` | HF hub, auto-downloaded |
| VLM model | `/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct` | **local snapshot** — download via `huggingface_hub.snapshot_download` (command is in the suite comments) |

The RL prompts and SFT parquet are generated automatically on first run (the
suite calls `tests/prepare_e2e_data.py` if they are missing); pass `--force` to
regenerate. Only the VLM snapshot must be downloaded manually. Everything else
is HF-hub auto-download.

---

## Step 9 — Smoke test and run the suite

```bash
# GPU-free unit tests first
python -m pytest tests/test_vllm_device_env.py tests/test_gloo_weight_sync.py -v

# multi-XPU E2E suite (exports the env vars from Step 7 itself)
bash tests/test_e2e_suite_multigpu.sh

# single-XPU subset
bash tests/test_e2e_suite_singlegpu.sh
```

Expected on XPU startup:
```text
vLLM weight-sync backend: gloo (CPU staging)
```

The suite is configured for **2 XPUs** (1 actor GPU + 1 vLLM GPU). To run on 3:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2
ray start --head --num-gpus=3
# actor on 2 GPUs + vLLM on 1:
#   --actor.num_gpus_per_node 2
#   --vllm.num_engines 1 --vllm.tensor_parallel_size 1
#   --train.colocate_actor_ref            # ref/reward share actor GPUs, no extra device
```

> The 2-GPU layout was validated on B70. The 3-GPU split is the documented recipe
> but was not run on 3× B60 here — validate it and watch for `grad_accum=0`
> (train batch size must be ≥ actor GPU count).

---

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `torch.xpu.is_available()` False | driver / groups / `/dev/dri` | re-map `/dev/dri`, add render/video groups |
| torch becomes `+cuNNN` after vLLM | vLLM pulled CUDA torch | rebuild vLLM with `--no-build-isolation`; reinstall Step 4 |
| `no 0.27.2 XPU wheel on PyPI` | it's a source build, not a release | build from the commit in Step 5 |
| actor + vLLM both on `xpu:0` | Ray overrode the selector | set `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` |
| `_xpu_C` cannot dlopen `.so` | kernels dir not on loader path | add `vllm_xpu_kernels` dir to `LD_LIBRARY_PATH` |
| `undefined symbol: urDeviceWaitExp` | base conda `libur_loader` shadows the venv's | put `$VIRTUAL_ENV/lib` first on `LD_LIBRARY_PATH` (Step 2) |
| every Ray placement group hangs on `{'GPU': 1.0}` | no `dpctl`, so Ray sees `GPU=0` | `ray start --head --num-gpus 2` |
| `Offloading is supported only for DeepSpeed FusedAdam` | `--ds.enable_sleep` + no `icpx` | fixed on this branch; see [ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md) |
| ZeRO-3 SIGSEGV in `ccl_worker_func` / `urEventGetInfo` | `overlap_comm` silently `True` at stage 3 | fixed on this branch; see [ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md) |
| weight sync silently stale | NCCL path selected on XPU | confirm startup logs `gloo`; don't pass `--vllm.sync_backend nccl` |
| XCCL hang / segfault | oneCCL/Level-Zero issues on B70 | use gloo; see [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md) |

---

## Notes

- **gloo is the supported default on XPU.** XCCL is experimental — see
  [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md).
- **Validated hardware is 2× Arc Pro B70, on torch 2.13.0+xpu.** On B70, torch 2.13
  still needs CCL env workarounds (Intel's "no env vars" claim was on B60). The gloo
  path is hardware-neutral and works on B60/B70 alike.
- **ZeRO stages 0-3 all pass on this stack** with the fixes on this branch, including
  with DeepSpeed sleep and vLLM sleep enabled — see
  [ZERO3_DEEPSPEED_SLEEP_FIXES.md](ZERO3_DEEPSPEED_SLEEP_FIXES.md) for the topology
  matrix and how to run its regression tests.
- On the 2× B70 box there is no direct PCIe P2P, so gloo (and XCCL) are
  host-staged — a hardware property, not a config error.
- The test suite uses `Qwen/Qwen2.5-0.5B` for speed; production runs use the
  larger models in `examples/scripts/`.
