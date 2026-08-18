# OpenRLHF on Intel XPU — Install Guide (verified `venv-tf` stack)

This guide reproduces the **exact environment that was validated end-to-end**
on Intel Arc Pro B70. Every version below is read from the working validation
environment (`venv-tf`), not inferred and not a PyPI guess.

> **Do not treat vLLM as a PyPI release.** The validated vLLM is an editable
> source build from a specific commit. There is no `0.23.1rc1` wheel on PyPI —
> that string is a source-build self-version. See Step 5.

---

## Verified stack (exact, from `venv-tf`)

| Component | Exact version | Source |
|---|---|---|
| Hardware | 2× Intel Arc Pro B70 (Battlemage, PCIe, no Xe Link) | — |
| Python | 3.13.12 | miniforge base |
| PyTorch | `torch==2.12.0+xpu` | pytorch.org XPU index |
| torchvision | `0.27.0+xpu` | same |
| torchaudio | `2.11.0+xpu` | same |
| triton | `triton-xpu==3.7.1` | same |
| vLLM | `0.23.1rc1.dev1200+g3e90d015b` | **editable source build** (Step 5) |
| vllm-xpu-kernels | `v0.1.11` wheel | GitHub release (Step 5) |
| Ray | `ray==2.55.0` | PyPI |
| DeepSpeed | `deepspeed==0.19.1` | PyPI |
| transformers | `5.14.1` | PyPI |
| oneCCL | `oneccl==2021.17.2`, `oneccl-devel==2021.17.2` | PyPI |
| Intel runtime | `intel-sycl-rt / -cmplr-lib-rt / -ur / -lic-rt / -opencl-rt / -openmp == 2025.3.2` | oneAPI 2025.3 |
| intel-pti | `0.16.0` | PyPI |
| numpy | `2.4.4` | PyPI |

Not used on XPU (see notes): **flash-attn is NOT installed** (CUDA-only).
`bitsandbytes==0.49.2` is present but unused — do not let it pull in a CUDA torch.

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

> If your base image ships torch **2.11**, you MUST replace it with **2.12.0+xpu**
> in Step 4. The validated stack is 2.12, not 2.11.

---

## Step 2 — oneAPI 2025.3 on PATH

The validated env used the oneAPI 2025.3 compiler. Put it first on PATH:

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
which icpx      # /opt/intel/oneapi/compiler/2025.3/bin/icpx
```

The Intel runtime pip packages (`intel-sycl-rt==2025.3.2` etc.) are pulled in
automatically by the torch-xpu wheels in Step 4 — do not install them separately.

---

## Step 3 — Python environment (3.13, matching validated)

```bash
conda create -n openrlhf-xpu python=3.13 -y
conda activate openrlhf-xpu
```

> The validated env is Python **3.13.12**. If your base image pins 3.12, that is
> usually fine, but 3.13 is the exercised version.

---

## Step 4 — Install PyTorch 2.12.0+xpu (exact companions)

```bash
pip install \
  torch==2.12.0+xpu \
  torchvision==0.27.0+xpu \
  torchaudio==2.11.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu

pip install triton-xpu==3.7.1
```

Verify XPU is live and torch is the XPU build:

```bash
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available(), 'count:', torch.xpu.device_count())"
# expect: 2.12.0+xpu  /  xpu: True  count: 2
```

If `xpu: False`: re-check `/dev/dri` mapping, render/video groups, and PATH.

---

## Step 5 — Build vLLM for XPU from the exact commit

There is **no wheel** — the validated vLLM was built editable from source at a
pinned commit, with `VLLM_TARGET_DEVICE=xpu`, against the torch 2.12 already
installed in Step 4.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3e90d015ba5bbad6f229ddb4c1f467ca33101524   # == v0.23.1rc0-1200-g3e90d015b

# --no-build-isolation forces the build to use the XPU torch already in the venv
# instead of pulling a fresh CUDA torch into an isolated build environment.
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation
```

Install the matching XPU kernels wheel:

```bash
pip install "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.11/vllm_xpu_kernels-0.1.11-cp38-abi3-manylinux_2_28_x86_64.whl"
```

If `_xpu_C` fails to `dlopen` a kernel `.so` at runtime, add the
`vllm_xpu_kernels` install directory to `LD_LIBRARY_PATH`.

Verify vLLM did not disturb torch:

```bash
python -c "import torch, vllm; print('torch', torch.__version__); print('vllm', vllm.__version__)"
# expect: torch 2.12.0+xpu  /  vllm 0.23.1rc1.dev1200+g3e90d015b
```

If torch now reads `+cuNNN`, vLLM pulled a CUDA build — reinstall Step 4 and
rebuild vLLM with `--no-build-isolation`.

---

## Step 6 — Install OpenRLHF (this fork, XPU branch)

```bash
git clone https://github.com/Madhustat/OpenRLHF.git
cd OpenRLHF
git checkout openrlfh_exp_multi

# --no-deps so the install does NOT pull a CUDA torch or a different vLLM
# over the XPU builds from Steps 4-5.
pip install -e . --no-deps

# then install the remaining pure-Python deps
pip install "ray==2.55.0" deepspeed==0.19.1 transformers==5.14.1
```

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
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$VENV/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1          # both XPUs visible
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 # else actor + vLLM collide on xpu:0
export OPENRLHF_WEIGHT_PROBE=0
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
```

`RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` is **mandatory** on XPU —
without it Ray overrides the device selector and the DeepSpeed actor and the
vLLM engine both land on `xpu:0`.

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
| `no 0.23.1rc1 on PyPI` | it's a source build, not a release | build from the commit in Step 5 |
| actor + vLLM both on `xpu:0` | Ray overrode the selector | set `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` |
| `_xpu_C` cannot dlopen `.so` | kernels dir not on loader path | add `vllm_xpu_kernels` dir to `LD_LIBRARY_PATH` |
| weight sync silently stale | NCCL path selected on XPU | confirm startup logs `gloo`; don't pass `--vllm.sync_backend nccl` |
| XCCL hang / segfault | torch 2.12 XCCL bugs | use gloo; see [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md) |

---

## Notes

- **gloo is the supported default on XPU.** XCCL is experimental and blocked on
  torch 2.13 + a matching vLLM build — see [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md).
- **Validated hardware is 2× Arc Pro B70.** On B70, torch 2.13 still needs CCL
  env workarounds (Intel's "no env vars" claim was on B60). The gloo path is
  hardware-neutral and works on B60/B70 alike.
- On the 2× B70 box there is no direct PCIe P2P, so gloo (and XCCL) are
  host-staged — a hardware property, not a config error.
- The test suite uses `Qwen/Qwen2.5-0.5B` for speed; production runs use the
  larger models in `examples/scripts/`.
