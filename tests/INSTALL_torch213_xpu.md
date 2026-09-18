# Install guide — OpenRLHF on Intel XPU (torch 2.13 stack)

This is the **actual, reproduced** install sequence used to validate the single-XPU
test suites — including the gotchas that a clean "pip install" misses. Every version
and step here was executed on the box (dut7054), not inferred.

Validated stack: Python 3.13.14 · torch 2.13.0+xpu · triton-xpu 3.7.2 ·
vLLM 0.27.1+xpu (source build @6e448d0ea) · vllm-xpu-kernels 0.1.12 ·
**transformers 5.16.0** · **kernels 0.16.2** · tokenizers 0.23.2 ·
**torchvision 0.28.0+cpu** · deepspeed 0.19.1 · ray 2.55.0 · mpi4py 4.1.2 · numpy.

Tooling note: this box uses **`uv`** (Python 3.13 fetched by uv). `uv`-created venvs have
**no `pip`** — use `uv pip install ...`. Substitute plain `pip` if you use a conda/venv
that has it.

---

## Step 1 — venv + PyTorch XPU

```bash
uv venv --python 3.13 /home/dut7054/madhu/venv-torch213-xpu
VENV=/home/dut7054/madhu/venv-torch213-xpu

uv pip install --python "$VENV/bin/python" torch==2.13.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
# triton-xpu 3.7.2 comes in with torch. Verify:
"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.xpu.device_count())"
# expect: 2.13.0+xpu  1   (no libsycl error when using a uv venv — no conda-lib conflict)
```

## Step 2 — runtime deps + vLLM XPU kernels

```bash
uv pip install --python "$VENV/bin/python" \
  numpy ray==2.55.0 deepspeed==0.19.1 \
  cmake ninja "setuptools-scm>=8" wheel packaging pybind11
uv pip install --python "$VENV/bin/python" \
  "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.12/vllm_xpu_kernels-0.1.12-cp38-abi3-manylinux_2_28_x86_64.whl"
```

## Step 3 — build vLLM from source (XPU) — the fiddly part

```bash
git clone https://github.com/vllm-project/vllm.git vllm-src-213 && cd vllm-src-213
git checkout 6e448d0ea

# vLLM's build-system requires setuptools_rust etc.; install them (no-build-isolation
# reuses the XPU torch already in the venv instead of pulling a CUDA torch):
uv pip install --python "$VENV/bin/python" setuptools_rust semantic-version jinja2

export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:$LD_LIBRARY_PATH"
VLLM_TARGET_DEVICE=xpu uv pip install --python "$VENV/bin/python" -e . --no-build-isolation
cd ..
```

### GOTCHA 1 — vLLM drags in CUDA companions that break the XPU stack
The vLLM install pulls **`torchvision`, `torchaudio`, `torchcodec`, and plain `triton 3.8.0`**.
Plain `triton` shadows `triton-xpu` (its Intel backend disappears:
`cannot import name 'intel'`), and CUDA `torchvision` is ABI-incompatible with torch-2.13+xpu
(`operator torchvision::nms does not exist`). **Fix — remove them and restore triton-xpu:**

```bash
uv pip uninstall --python "$VENV/bin/python" torchvision torchaudio torchcodec triton
uv pip install --python "$VENV/bin/python" --reinstall triton-xpu==3.7.2 \
  --index-url https://download.pytorch.org/whl/xpu
```

### GOTCHA 2 — vLLM/VLM still need torchvision at runtime (use the +cpu build)
vLLM's warmup path and **Qwen2-VL's `Qwen2VLVideoProcessor` import torchvision**
(`ImportError: ... requires the Torchvision library`). The **CPU-index** build matches
torch-2.13's ABI and imports cleanly (`nms` works); the default/CUDA one does not.

```bash
uv pip install --python "$VENV/bin/python" torchvision==0.28.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -c "from torchvision.ops import nms; print('torchvision nms OK')"
```

## Step 4 — transformers / kernels (packing needs >=5.16)

```bash
# transformers 5.7 fails sample-packing (flash-attn2/torch-2.13 kernel gap). Use 5.16:
uv pip install --python "$VENV/bin/python" --no-deps transformers==5.16.0
uv pip install --python "$VENV/bin/python" --no-deps "tokenizers>=0.23.1,<0.24.0"  # 5.16 needs this
uv pip install --python "$VENV/bin/python" "kernels>=0.16.0,<0.17.0"                # 0.16.2
uv pip install --python "$VENV/bin/python" mpi4py                                   # supervised trainers
```

## Step 5 — OpenRLHF deps (exclude torch + flash-attn)

```bash
# Do NOT `pip install -r requirements.txt` directly: it would pull a CPU/CUDA torch
# (clobbering the XPU build) and try to build flash-attn (CUDA-only). Install the rest:
uv pip install --python "$VENV/bin/python" \
  accelerate aiohttp bitsandbytes datasets einops "grpcio>=1.74.0" "huggingface_hub>=1.0.0" \
  isort jsonlines loralib optimum "optree>=0.15.0" packaging peft pylatexenc "pynvml>=12.0.0" \
  "ray[default]==2.55.0" sympy tensorboard torchdata torchmetrics tqdm \
  transformers_stream_generator wandb wheel
# flash-attn is CUDA-only; OpenRLHF's ring_attn_utils has an optional fallback, so skip it.
# OpenRLHF itself is used via PYTHONPATH (not pip-installed) — see Step 6.
```

## Step 6 — environment (export before running)

```bash
export VIRTUAL_ENV=/home/dut7054/madhu/venv-torch213-xpu
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"   # venv lib FIRST
export PYTHONPATH="/path/to/OpenRLHF-checkout:$PYTHONPATH"   # openrlhf not pip-installed
export ONEAPI_DEVICE_SELECTOR=level_zero:0                   # pin to one XPU
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1       # else actor+vLLM collide on xpu:0
export OPENRLHF_DS_TORCH_ADAM=1                              # no icpx JIT -> torch AdamW
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
```

Ray has no `dpctl` here, so it can't auto-discover XPUs — **start the head with the GPU
count declared explicitly**, or placement groups hang:
```bash
ray start --head --num-gpus=1
```

## Verify

```bash
"$VENV/bin/python" - <<'PY'
import torch, vllm, transformers, torchvision, deepspeed, mpi4py
print("torch", torch.__version__, "| xpu", torch.xpu.device_count())
print("vllm", vllm.__version__, "| transformers", transformers.__version__,
      "| torchvision", torchvision.__version__)
from vllm.platforms import current_platform  # -> XPUPlatform
PY
```

## Gotcha summary (what bites you, and the fix)

| Symptom | Fix |
|---|---|
| `No module named pip` in the venv | use `uv pip install` |
| vLLM build: `No module named setuptools_rust` | install vLLM build-system reqs first (`setuptools_rust`, `jinja2`, …) |
| `cannot import name 'intel' from triton._C` | plain `triton` shadowed `triton-xpu` — uninstall `triton`, reinstall `triton-xpu==3.7.2` from the XPU index |
| `operator torchvision::nms does not exist` / `Qwen2VLVideoProcessor requires Torchvision` | install **`torchvision==0.28.0+cpu`** from the CPU index (matches torch-2.13 ABI) |
| Sample-packing: `Cannot find a build variant … torch213 … flash-attn2` | upgrade to **transformers 5.16.0** (+ tokenizers 0.23.2, kernels 0.16.2) |
| torch becomes `+cpu`/`+cuXXX` after some install | you let a dep pull a non-XPU torch — reinstall `torch==2.13.0+xpu`, use `--no-deps` for transformers |
| Ray placement group hangs on `{'GPU': 1.0}` | no `dpctl` → `ray start --head --num-gpus=N` |
| Supervised trainers: `No module named mpi4py` | `uv pip install mpi4py` |
