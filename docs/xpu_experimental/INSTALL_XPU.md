# OpenRLHF on Intel XPU — Fresh Environment Install Guide

Verified stack (2× Intel Arc Pro B70, Battlemage, PCIe, no Xe Link):

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 (or 22.04) |
| Intel GPU driver | Latest LTS from intel-graphics PPA |
| oneAPI Base Toolkit | 2025.3 |
| Python | 3.10 – 3.11 |
| PyTorch | 2.12.0+xpu |
| oneCCL | 2021.17 (bundled with torch-xpu) |
| vLLM | 0.23.1rc1 (built for XPU) |
| Ray | 2.55.0 |

> The exact torch / vLLM pairing matters. Do not mix a CUDA torch with an
> XPU vLLM. Everything below installs the XPU build consistently.

---

## Step 0 — Verify hardware and driver

```bash
# GPUs visible to the kernel
lspci | grep -i display

# Intel GPU tooling (install if missing: sudo apt install intel-gpu-tools)
sudo xpu-smi discovery      # or: clinfo | grep "Device Name"
```

You should see both Arc Pro B70 devices listed.

---

## Step 1 — Install the Intel GPU driver (host)

Skip this if you run inside Intel's prebuilt Docker image (recommended — see
Step 6). Only needed for bare-metal installs.

```bash
# Add Intel graphics repo (Ubuntu 24.04 example)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

sudo apt update
sudo apt install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1

# add your user to render/video groups
sudo usermod -aG render,video $USER
# log out / back in for group change to take effect
```

---

## Step 2 — Install oneAPI Base Toolkit

```bash
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/<installer>.sh
sudo sh ./<installer>.sh   # select Base Toolkit; 2025.3 recommended

# every shell that runs training must source this:
source /opt/intel/oneapi/setvars.sh
```

Verify the compiler is on PATH:
```bash
which icpx      # /opt/intel/oneapi/compiler/2025.3/bin/icpx
```

---

## Step 3 — Create a clean Python environment

```bash
# conda / miniforge recommended
conda create -n openrlhf-xpu python=3.11 -y
conda activate openrlhf-xpu
```

---

## Step 4 — Install PyTorch for XPU

```bash
pip install torch==2.12.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/xpu
```

Verify XPU is visible to torch:
```bash
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available(), 'count:', torch.xpu.device_count())"
# expect: 2.12.0+xpu  /  xpu: True  count: 2
```

If `xpu: False`: re-source `setvars.sh`, confirm you are in the render/video
groups, and re-check the driver from Step 1.

---

## Step 5 — Install vLLM (XPU build) and OpenRLHF

```bash
# vLLM must be the XPU build matching torch 2.12 — build from source per
# vLLM's XPU instructions, or install the prebuilt XPU wheel if available:
#   https://docs.vllm.ai/en/latest/getting_started/installation/xpu.html

# OpenRLHF from this fork (XPU-enabled branch)
git clone https://github.com/Madhustat/OpenRLHF.git
cd OpenRLHF
git checkout openrlfh_exp_multi     # or the specific PR branch you need
pip install -e .                    # installs Ray, DeepSpeed, transformers, etc.
```

Pin Ray if the resolver drifts:
```bash
pip install "ray==2.55.0"
```

---

## Step 6 — (Recommended) Use Intel's prebuilt Docker image

The bare-metal path (Steps 1–2) is fragile. Intel's XPU PyTorch image ships
the driver + oneAPI + torch-xpu preconfigured:

```bash
docker run -it --rm \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  --shm-size=16g \
  -v $HOME/OpenRLHF:/workspace/OpenRLHF \
  intel/intel-extension-for-pytorch:2.12.0-xpu \
  /bin/bash

# inside container:
cd /workspace/OpenRLHF && pip install -e .
```

`--device /dev/dri` and the `render`/`video` groups are mandatory — without
them torch will report `xpu: False`.

---

## Step 7 — Source the XPU runtime env before training

OpenRLHF weight sync uses **gloo** on XPU by default (no extra env needed).
The env vars below are only required for the experimental **XCCL** path.
See [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md) for why each is needed.

```bash
# gloo path (default, production) — no special CCL env required
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1   # make both XPUs visible

# --- only if using experimental XCCL backend ---
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
export FI_PROVIDER=shm
export CCL_ZE_CACHE_OPEN_IPC_HANDLES=0
export CCL_ZE_CACHE_GET_IPC_HANDLES=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
```

---

## Step 8 — Smoke test

```bash
# 1. tests that don't need a GPU
python -m pytest tests/test_vllm_device_env.py tests/test_gloo_weight_sync.py -v

# 2. minimal single-XPU GRPO run (colocated)
cd OpenRLHF
bash tests/test_e2e_suite_singlegpu.sh     # runs the single-GPU subset
```

Expected on XPU startup:
```text
vLLM weight-sync backend: gloo (CPU staging)
```

If you see that line and the run reaches training steps with moving rewards,
the environment is correctly set up.

---

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `torch.xpu.is_available()` is False | driver / groups / setvars | re-source setvars, add render/video groups, `--device /dev/dri` in Docker |
| `ZE_AFFINITY_MASK` conflict / wrong device | `ONEAPI_DEVICE_SELECTOR` set alongside it | OpenRLHF clears the selector before pinning; don't set both manually |
| vLLM import errors | CUDA vLLM with XPU torch | reinstall the XPU vLLM build |
| Weight sync silently stale | NCCL path selected on XPU | confirm startup logs `gloo`; do not pass `--vllm.sync_backend nccl` |
| XCCL hangs / segfaults | torch 2.12 XCCL bugs | use gloo; see [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md) |

---

## Notes

- **gloo is the supported default on XPU.** XCCL is experimental and blocked on
  torch 2.13 + a matching vLLM build — see [XCCL_KNOWN_ISSUES.md](XCCL_KNOWN_ISSUES.md).
- On the 2× B70 box there is no direct PCIe P2P, so both gloo and XCCL are
  host-staged. This is a hardware property, not a config error.
- The test suite uses `Qwen/Qwen2.5-0.5B` for speed; production runs use the
  larger models in `examples/scripts/`.
