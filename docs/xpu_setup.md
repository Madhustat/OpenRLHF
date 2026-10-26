# OpenRLHF on Intel XPU: requirements and settings

This page lists everything to install and set before running any model on Intel XPU (Intel Arc Pro B70) with this
branch, and how weight sync between the trainer and vLLM works. It describes the native (no container) environment the
branch was validated on in October 2026.

The branch is upstream main `dc2a7ad` plus:
- PRs #1267, #1268, #1269, #1301, #1302 and #1363;
- vLLM weight sync on Intel XPU over XCCL (direct XPU-to-XPU) or gloo (through the CPU).

In the commands below, `V` is the Python environment and `R` the OpenRLHF checkout. Change both for your machine:

```bash
export V=$HOME/venv_openrlhf_xpu
export R=$HOME/OpenRLHF
export ICPX_DIR=/opt/intel/oneapi/compiler/2026.1/bin
```

## 1. Host requirements

| Item | Validated version | Notes |
|---|---|---|
| OS, kernel | Ubuntu 24.04.2, kernel 7.0.0-38-generic | the in-kernel `xe` driver |
| GPUs | 2x Intel Arc Pro B70 | |
| Host RAM | 125 GB | PPO with a critic and `--ds.adam_offload` on 2 trainer ranks needs most of it |
| GuC firmware | 70.44.1 (`xe/bmg_guc_70.bin`) | loaded by the kernel |
| Compute runtime | `libze-intel-gpu1`, `intel-opencl-icd` 26.31.39395.14 | Ubuntu PPA `ppa:kobuk-team/intel-graphics` |
| IGC | `libigc2` 2.41.9 | same PPA |
| Level Zero loader | `libze1` 1.32.0 | |
| xpu-smi | 2.1.0 | health and memory checks |
| oneAPI DPC++/C++ compiler | `icpx` 2026.1.1 | Only `icpx` on `PATH` is needed (Triton XPU, DeepSpeed JIT ops). Do not source `setvars.sh`: it puts the system oneCCL on the library path |
| Python | 3.12.3 | |
| GPU access | your user in the `render` and `video` groups | |
| Network | Hugging Face Hub reachable | models, datasets and the flash-attn2 Hub kernel download on first use |

Check: `xpu-smi discovery` lists the GPUs.

## 2. Python packages

| Component | Version |
|---|---|
| PyTorch | 2.14.0+xpu (torchvision 0.29.1+xpu, torchaudio 2.11.0+xpu) |
| vLLM | 0.32.0+xpu with the one-line fix of section 3.5; vllm-xpu-kernels 0.1.15.4 |
| oneCCL | 2022.1.2 (pip `oneccl`, `oneccl-devel`) |
| Intel runtimes | intel-sycl-rt, intel-cmplr-lib-rt, -lib-ur, -lic-rt, dpcpp-cpp-rt, intel-opencl-rt, intel-openmp 2026.1.1; intel-pti 1.1.0; impi-rt 2021.18.1 |
| Triton | triton 3.8.0+xpu, triton-xpu 3.8.0 |
| DeepSpeed | 0.19.8 from git `dcd8595c` (0.19.6 does not build for torch 2.14) |
| transformers | 5.19.0 (vLLM 0.32 needs >=5.16.1,<5.20) |
| Others | ray 2.55.0, accelerate 1.14.0, peft 0.20.0, bitsandbytes 0.50.2, datasets 5.1.0, huggingface_hub 2.2.0, tokenizers 0.23.3, numpy 2.3.5, kernels 0.17.2, mpi4py 4.1.2 |
| flash-attn | not installed (CUDA-only). On XPU `flash_attention_2` comes from the Hub kernel `kernels-community/flash-attn2` through `kernels` |

## 3. Build the environment

Use a shell without a `LD_LIBRARY_PATH` from another Python stack (for example miniforge): an older
`libur_loader`/`libsycl`/`libccl` there makes `import torch` fail with `undefined symbol: urDeviceWaitExp`.

```bash
unset LD_LIBRARY_PATH CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT
```

### 3.1 venv and vLLM

```bash
python3.12 -m venv $V && $V/bin/pip install uv
UV="$V/bin/uv pip install --python $V/bin/python"
$UV 'vllm==0.32.0+xpu' --extra-index-url https://wheels.vllm.ai/0.32.0/xpu \
    --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
```

### 3.2 Intel runtimes

```bash
$UV 'intel-cmplr-lib-rt==2026.1.1' 'intel-cmplr-lib-ur==2026.1.1' 'intel-cmplr-lic-rt==2026.1.1' \
    'intel-sycl-rt==2026.1.1' 'oneccl-devel==2022.1.2' 'oneccl==2022.1.2' 'dpcpp-cpp-rt==2026.1.1' \
    'intel-opencl-rt==2026.1.1' 'intel-openmp==2026.1.1' 'intel-pti==1.1.0'
```

### 3.3 OpenRLHF's requirements, keeping vLLM's stack

Create three files.

`requirements_openrlhf_xpu.txt` (OpenRLHF's `requirements.txt` without deepspeed, flash-attn, torch and transformers):
```
accelerate==1.14.0
aiohttp
bitsandbytes==0.50.2
datasets
einops
grpcio>=1.74.0
huggingface_hub>=1.0.0
isort
jsonlines
loralib
optimum
optree>=0.15.0
packaging
peft==0.20.0
pylatexenc
pynvml>=12.0.0
ray[default]==2.55.0
sympy
tensorboard
torchdata
torchmetrics
tqdm
transformers_stream_generator
wandb
wheel
pytest
```

`constraints_vllm_core.txt`:
```
auto-round-lib==0.16.0
compressed-tensors==0.17.0
huggingface_hub==2.2.0
numpy==2.3.5
outlines_core==0.2.14
safetensors==0.8.0
tokenizers==0.23.3
torch==2.14.0+xpu
torchaudio==2.11.0+xpu
torchcodec==0.17.0+cpu
torchvision==0.29.1+xpu
transformers==5.19.0
triton==3.8.0+xpu
triton-xpu==3.8.0
vllm==0.32.0+xpu
vllm-xpu-kernels==0.1.15.4
xgrammar==0.2.8
```

`overrides_oneapi.txt`:
```
dpcpp-cpp-rt==2026.1.1
impi-rt==2021.18.1
intel-cmplr-lib-rt==2026.1.1
intel-cmplr-lib-ur==2026.1.1
intel-cmplr-lic-rt==2026.1.1
intel-opencl-rt==2026.1.1
intel-openmp==2026.1.1
intel-pti==1.1.0
intel-sycl-rt==2026.1.1
mkl==2026.1.0
oneccl==2022.1.2
oneccl-devel==2022.1.2
onemkl-license==2026.1.0
onemkl-sycl-blas==2026.1.0
onemkl-sycl-dft==2026.1.0
onemkl-sycl-lapack==2026.1.0
onemkl-sycl-rng==2026.1.0
onemkl-sycl-sparse==2026.1.0
tbb==2023.1.0
tcmlib==1.5.0
umf==1.1.0
```

```bash
$UV -c constraints_vllm_core.txt --override overrides_oneapi.txt -r requirements_openrlhf_xpu.txt \
    --extra-index-url https://wheels.vllm.ai/0.32.0/xpu --extra-index-url https://download.pytorch.org/whl/xpu \
    --index-strategy unsafe-best-match
```

### 3.4 DeepSpeed, `kernels` and `mpi4py`

```bash
env -u LD_LIBRARY_PATH $V/bin/python -m pip install --no-deps \
    'git+https://github.com/deepspeedai/DeepSpeed@dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c'
$UV -c constraints_vllm_core.txt --override overrides_oneapi.txt hjson \
    --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
$V/bin/python -m pip list --format=freeze | grep -E '^[A-Za-z0-9_.-]+==[^ ]+$' > constraints_existing.txt
$V/bin/python -m pip install -c constraints_existing.txt 'kernels>=0.17.0,<0.18.0' 'mpi4py==4.1.2'
```

- DeepSpeed must be built with build isolation and without `LD_LIBRARY_PATH`. Expect
  `Successfully installed deepspeed-0.19.8+dcd8595c`.
- `kernels` is needed by `flash_attention_2`, which `--train.dynamic_batch_enable` and `--ds.packing_samples` use.
- `mpi4py` is needed when `train_sft`, `train_rm` or `train_dpo` start with plain `python -m`.

### 3.5 vLLM fix: GroupCoordinator `device_ids` on XPU

vLLM 0.32.0 places its XPU communication groups by local rank instead of by `device_ids`. Without this fix, a vLLM
engine on 2+ XPUs in non-sequential order (for example TP=2 on XPUs `[1,0]`) fails in send/recv. TP=1 engines work
without it.

```bash
cd $V/lib/python3.12/site-packages && patch -p1 <<'EOF'
--- a/vllm/distributed/parallel_state.py
+++ b/vllm/distributed/parallel_state.py
@@ -535,7 +535,12 @@
             )
             self.device = torch.device(f"cuda:{visible_device_index}")
         elif current_platform.is_xpu():
-            self.device = torch.device(f"xpu:{self.device_index}")
+            visible_device_index = (
+                current_platform.logical_device_id_to_visible_device_id(
+                    self.device_index
+                )
+            )
+            self.device = torch.device(f"xpu:{visible_device_index}")
         elif current_platform.is_out_of_tree():
             self.device = torch.device(
                 f"{current_platform.device_name}:{self.device_index}"
EOF
```

## 4. Get the code

```bash
git clone https://github.com/Madhustat/OpenRLHF.git $R && cd $R && git checkout xpu/openrlhf_vllm032_allprs_lean
```

OpenRLHF runs from the checkout through `PYTHONPATH`; it is not pip-installed.

## 5. Settings for every shell

Save as `env_xpu.sh` and `source` it before any command, including `ray start`:

```bash
unset CCL_ROOT FI_PROVIDER_PATH I_MPI_ROOT ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR
export LD_LIBRARY_PATH=$V/lib
source $V/bin/activate
export PATH="$PATH:$ICPX_DIR"
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=8589934592 CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=8589934592
export TORCH_EXTENSIONS_DIR=$HOME/torch_extensions_openrlhf_xpu
export KERNELS_CACHE=$HOME/hf_kernels_cache
export PYTHONPATH=$R
```

| Setting | Why |
|---|---|
| `LD_LIBRARY_PATH=$V/lib` | the venv's libraries only; the pip oneCCL loads its plugins from there |
| `icpx` on `PATH` | Triton XPU and DeepSpeed's JIT ops compile with it |
| `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` | required: every process sees every XPU and is pinned to its Ray-assigned one by index. OpenRLHF stops at start without it |
| the two `CCL_SYCL_*_SIMPLE_THRESHOLD` | required: with oneCCL's defaults, a large all-reduce/all-gather between two XPUs keeps the peer's freed GPU memory mapped until the process exits (about 2 GB per training step in a PPO run with 2 trainer ranks). With these settings that memory stays flat |
| `TORCH_EXTENSIONS_DIR` | a JIT cache for this stack only |
| `KERNELS_CACHE` | a Hub-kernel cache you own; a root-owned lock file in a shared cache breaks every packing / dynamic-batch run |

## 6. Verify

```bash
source env_xpu.sh && cd /tmp
python -c "import torch, vllm, deepspeed; print(torch.__version__, vllm.__version__, deepspeed.__version__)"
#   2.14.0+xpu 0.32.0 0.19.8+dcd8595c
python -c "import torch; print(torch.xpu.device_count(), torch.distributed.is_xccl_available())"
#   2 True
cd $R && HF_HUB_OFFLINE=1 python -m pytest -q -p no:cacheprovider tests/
#   246 passed, 2 skipped
journalctl -k --since -10min | grep -iE 'xe .*(reset|wedged|hang|fault|error)'
#   (nothing)
```

## 7. Run

```bash
source env_xpu.sh
ray start --head --num-gpus 2 --disable-usage-stats
python -m openrlhf.cli.train_ppo_ray ...
```

### Weight sync between the trainer and vLLM

| `--vllm.sync_backend` | What happens |
|---|---|
| not given (`auto`) | If vLLM shares the trainer's XPU (`--train.colocate_all`): **gloo**. If vLLM runs on XPUs of its own: **XCCL**, after the startup test below |
| `gloo` | always gloo, with any number of XPUs |
| `xccl` | always the XCCL startup test, then XCCL. If vLLM shares the trainer's XPU, it logs an "experimental" warning first |
| `nccl` | refused on XPU |

- **gloo** copies each weight trainer XPU -> CPU -> vLLM XPU.
- **XCCL** copies each weight directly XPU -> XPU.
- **XCCL startup test:** before any real weight, one 128 MiB broadcast from the trainer to every vLLM rank must arrive
  bit-for-bit within `--vllm.sync_timeout` (default 600 s). If it does not, the job stops with
  `XCCL weight sync failed its startup test ... Rerun with --vllm.sync_backend gloo`.
- **During training on XCCL,** each tensor's sync has the same time limit. A hang stops the job and prints every
  thread's stack.
- Nothing switches to gloo by itself.

**One XPU:** the trainer and vLLM have to share it, so run with `--train.colocate_all`; `auto` then uses gloo. Without
`--train.colocate_all` the trainer and each vLLM engine ask Ray for a whole XPU each, and the job waits for a second
XPU.

### Upstream `examples/scripts` on XPU

They are written for 8 NVIDIA GPUs:
- replace `--vllm.sync_backend nccl` with `auto` (or drop it);
- scale the model and the GPU counts to your XPUs;
- `flash_attention_2`, packing and dynamic batching use the Hub kernel (section 3.4); its first use needs Hub access.

### Containers

XCCL needs `pidfd_getfd()`, which Docker's default seccomp profile denies: run with `--cap-add SYS_PTRACE` (or
`--security-opt seccomp=unconfined`), or use `--vllm.sync_backend gloo`.

## 8. Known limits

| Case | What happens |
|---|---|
| ZeRO-3 with vLLM on the trainer's XPUs, in some layouts (for example 2 trainer ranks + critic, vLLM sleep off) | GPU fault (device lost) inside DeepSpeed's ZeRO-3 parameter all-gather. Other ZeRO-3 layouts pass |
| ZeRO-0 with `--ds.enable_sleep` | DeepSpeed's ZeRO-0 optimizer has no `offload_states`: unsupported |
| ZeRO-0 with `--vllm.enable_sleep`, vLLM at `--vllm.gpu_memory_utilization 0.22` | vLLM runs out of memory at wake-up; 0.15 works |
| ZeRO-2, 2 trainer ranks + critic, 2 vLLM engines sharing the XPUs, `--ds.enable_sleep` without `--vllm.enable_sleep` | vLLM's startup memory check fails: memory is freed on the same XPU while vLLM measures |
| `--ds.tensor_parallel_size 2` (DeepSpeed AutoTP) | both trainer ranks abort at step 0 |
| `flash_attention_2` on the trainer (`--ds.attn_implementation`) | one GPU page fault at step 5 in a 2-XPU run; not root-caused |
| `serve_remote_rm` with 2 XPUs visible | upstream issue #938 (`device_map="auto"` splits the model); pin it to one XPU |
| `train_rm` with fewer than 100 samples and no eval dataset | the eval split has 0 rows; use at least 100 samples |
| PPO + critic at TP=2 with 2 trainer ranks and `--ds.adam_offload` | host RAM can run out (Ray kills a worker) |
| Shared GPUs | another job on the same XPUs breaks runs. After any `reset` / `wedged` / `fault` kernel message, check the XPUs (section 6) before the next run |

## 9. Validated on this branch and environment (October 2026)

| Test | Result |
|---|---|
| Unit tests | 246 passed, 2 skipped |
| Weight sync end to end: XCCL, gloo, `auto` with vLLM on its own XPU and with vLLM sharing the trainer's XPU | all pass, every checked weight bit-identical in vLLM; a corrupted startup test and a hung one both stop the job as described |
| 17 upstream example cases on 2 XPUs | 16 pass; `serve_remote_rm` fails (#938) |
| 80-cell layout matrix (ZeRO-0 to 3, vLLM / DeepSpeed sleep, TP=1/TP=2, with and without a critic) | 54 pass, 8 not allowed by OpenRLHF, 9 unsupported by DeepSpeed (ZeRO-0 + DeepSpeed sleep), 1 fail, 2 GPU faults (ZeRO-3), 6 ZeRO-3 cells not run |
| 57-case extended suite (algorithms, reward models, MoE, SFT/RM/DPO, LoRA, checkpoints) | 48 pass; 7 not passing as before (2 need CUDA, 2 need packages that are not installed, 3 known non-XPU issues: weight sync through Ray collectives, 4-bit RL, RL determinism); 2 new failures (AutoTP, `flash_attention_2`) |
| GPU memory with the oneCCL settings | flat across steps (with oneCCL defaults: +2 GB per step until vLLM runs out of memory) |
| Speed with the oneCCL settings | run time unchanged: 7 upstream example cases each within 4 s of the run without them |

Not tested: NVIDIA for this branch; more than 2 XPUs; XCCL between processes sharing one XPU.
