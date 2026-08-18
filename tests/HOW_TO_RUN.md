# How to Run the OpenRLHF XPU Test Suite

This guide covers every step from a clean terminal to a full passing test run —
unit tests, the multi-GPU E2E suite, and the single-GPU E2E suite.

---

## Test suite overview

| Suite | File | Tests | Hardware | Time |
|---|---|---|---|---|
| Unit tests | `pytest tests/` | 86 | No GPU needed | ~40 s |
| Multi-GPU E2E | `test_e2e_suite_multigpu.sh` | 28 | 2 physical GPUs | ~2 hrs |
| Single-GPU E2E | `test_e2e_suite_singlegpu.sh` | 19 | 1 physical GPU | ~1 hr |

See `TESTS.md` for a full description of every test.

---

## Step 0 — Prerequisites

### Hardware
- This guide targets **2× Intel Arc Pro B70** (XPU).
- The single-GPU suite also runs on 1× B70.
- For NVIDIA, the same commands work — remove the XPU-specific env vars
  (Steps 1 and 2 below) and replace `--vllm.sync_backend gloo` with `nccl`.

### Software
| Component | Version used |
|---|---|
| PyTorch | 2.12.0+xpu |
| vLLM | 0.23.1rc1 (source build — see INSTALL_XPU.md) |
| Ray | 2.55.0 |
| oneAPI compiler | 2025.3 |

See [../docs/xpu_experimental/INSTALL_XPU.md](../docs/xpu_experimental/INSTALL_XPU.md)
to build this exact environment from scratch.

---

## Step 1 — Open a terminal and activate your environment

Activate whatever Python environment has the stack from INSTALL_XPU.md (conda
env or venv), then set the two XPU device variables. The suites find `python`
and `ray` on your `PATH`; override with `PYTHON=... RAY=...` if they live
elsewhere.

```bash
conda activate openrlhf-xpu          # or: source <your-venv>/bin/activate
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
```

**Why each one is needed:**
- `ONEAPI_DEVICE_SELECTOR` — makes both Arc Pro B70 GPUs visible (device 0 and device 1). Without this, only one may appear.
- `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR` — prevents Ray from overriding the device selector inside worker processes (Ray misdetects Arc Pro B70 as NVIDIA and would set the wrong variable).

The suites prepend the oneAPI 2025.3 compiler dir to `PATH` automatically if it
exists at `/opt/intel/oneapi/compiler/2025.3/bin` (needed for the DeepSpeed JIT).

**Verify the setup:**
```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.device_count(), 'XPU devices')"
# Expected output:
# 2.12.0+xpu
# 2 XPU devices
```

---

## Step 2 — Change to the repo directory

All test commands must be run from the repo root:

```bash
cd <your-clone>/OpenRLHF-multi
```

> **Datasets are automatic.** The RL prompts and SFT parquet are generated from
> GSM8K into `tests/data/` the first time you launch a suite (via
> `tests/prepare_e2e_data.py`). The first run needs network for the GSM8K
> download; after that it is cached. The only manual data step is the VLM model
> in Step 5 (used by one test).

---

## Step 3 — Run the unit tests

Unit tests require no GPU, no Ray cluster, and no model download.
They complete in about 40 seconds.

```bash
python -m pytest tests/ -q --ignore=tests/results
```

**Expected output:**
```
======================= 86 passed, 15 warnings in 37.09s =======================
```

**Run a specific test file:**
```bash
python -m pytest tests/test_gloo_weight_sync.py -v
python -m pytest tests/test_loss_aggregation.py -v
python -m pytest tests/test_ring_attn_utils.py -v
python -m pytest tests/test_ray_env_vars.py -v
python -m pytest tests/test_vllm_device_env.py -v
python -m pytest tests/test_distributed_backend_generic.py -v
```

---

## Step 4 — (Multi-GPU only) Check GPU health before starting

Before any E2E run, confirm both GPUs are free and responsive:

```bash
# Check both GPUs are detected
xpu-smi discovery | grep "Device Name"
# Expected: two lines showing "Intel(R) Arc(TM) Pro B70 Graphics"

# Check no stale processes are holding GPU memory
ps aux | grep -iE "train_ppo_ray|EngineCore|PolicyModelActor|RolloutRayActor" | grep -v grep
# Expected: no output (no leftover processes)

# Check Ray is stopped
ray stop --force 2>/dev/null
rm -rf /tmp/ray
```

If any stale training processes appear, kill them before running:
```bash
pkill -9 -f "train_ppo_ray|EngineCore|PolicyModelActor|RolloutRayActor|ray::"
sleep 3
rm -rf /tmp/ray
```

---

## Step 5 — (VLM test only) Download the VLM model

The multi-GPU suite includes a VLM test (`vlm_grpo`) that uses
`Qwen2-VL-2B-Instruct`. The HuggingFace model cache may be root-owned
from a previous Docker run. Download the model to a clean writable path first:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2-VL-2B-Instruct',
    local_dir='/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct'
)
print('Done')
"
```

This takes a few minutes and downloads ~4 GB. You only need to do it once;
the model stays at `/tmp/hf_vlm_clean/` until the machine reboots.

**After a reboot** — re-download before running the VLM test:
```bash
# Quick check: is it still there?
du -sh /tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct 2>/dev/null || echo "need to re-download"
```

---

## Step 6 — Run the E2E suites

### Option A: Multi-GPU suite (28 tests, ~2 hrs)

Requires 2 physical GPUs. Runs all algorithms, colocation modes, VLM, async,
DAPO, ProRL v2, and all supervised training paths.

```bash
# Run all 28 tests (recommended: use nohup so it survives disconnection)
nohup bash tests/test_e2e_suite_multigpu.sh > tests/multigpu_run.log 2>&1 &
echo "PID=$! — tail -f tests/multigpu_run.log to monitor"
```

**Monitor progress:**
```bash
# See which test is currently running
tail -f tests/multigpu_run.log

# See which tests have passed/failed so far
RDIR=$(ls -td tests/results/e2e_suite_* | head -1)
cat $RDIR/summary.txt
```

**Expected final output:**
```
────────────────────────────────────
TOTAL: 28 tests
  PASS: 28
  FAIL: 0
  SKIP: 0 (filtered)
────────────────────────────────────
```

---

### Option B: Single-GPU suite (19 tests, ~1 hr)

Requires 1 physical GPU. All RL tests use `colocate_all + enable_sleep`
so the actor and vLLM engine time-share the single GPU.

```bash
# For single-GPU: restrict to device 0 only
export ONEAPI_DEVICE_SELECTOR=level_zero:0

nohup bash tests/test_e2e_suite_singlegpu.sh > tests/singlegpu_run.log 2>&1 &
echo "PID=$! — tail -f tests/singlegpu_run.log to monitor"
```

**Monitor progress:**
```bash
tail -f tests/singlegpu_run.log

RDIR=$(ls -td tests/results/singlegpu_* | head -1)
cat $RDIR/summary.txt
```

**Expected final output:**
```
────────────────────────────────────
TOTAL: 19 tests
  PASS: 19
  FAIL: 0
  SKIP: 0 (filtered)
────────────────────────────────────
```

---

## Running a subset

Both suites accept an optional filter string. Any test whose ID contains
the filter runs; all others are skipped.

```bash
# Multi-GPU: run only GRPO tests
bash tests/test_e2e_suite_multigpu.sh grpo

# Multi-GPU: run only the baseline
bash tests/test_e2e_suite_multigpu.sh grpo_zero2_nocolo

# Multi-GPU: run only supervised tests
bash tests/test_e2e_suite_multigpu.sh sft
bash tests/test_e2e_suite_multigpu.sh dpo
bash tests/test_e2e_suite_multigpu.sh rm

# Single-GPU: run only RL tests
bash tests/test_e2e_suite_singlegpu.sh sg_grpo
bash tests/test_e2e_suite_singlegpu.sh sg_reinforce

# Single-GPU: run one test
bash tests/test_e2e_suite_singlegpu.sh sg_sft_zero2
```

---

## Reading the results

After a run, results are in `tests/results/`:

```
tests/results/
  e2e_suite_20260813T014319/   ← multi-GPU run (timestamp in name)
    summary.txt                ← PASS/FAIL/SKIP for every test
    grpo_zero2_nocolo.log      ← full output of each test
    grpo_zero3_nocolo.log
    ...
  singlegpu_20260813T080000/   ← single-GPU run
    summary.txt
    sg_grpo_baseline.log
    ...
```

**Check all results at once:**
```bash
# Multi-GPU: last run
cat $(ls -td tests/results/e2e_suite_* | head -1)/summary.txt

# Single-GPU: last run
cat $(ls -td tests/results/singlegpu_* | head -1)/summary.txt
```

**Dig into a failure:**
```bash
# View the full log for a failed test
RDIR=$(ls -td tests/results/e2e_suite_* | head -1)
cat $RDIR/grpo_zero2_nocolo.log | grep -E "Global step|Error|Traceback|EXIT"
```

---

## Troubleshooting

### Test hangs at vLLM model loading
Most likely a stale process from a previous run is holding GPU memory.

```bash
# Kill everything and clean up
ray stop --force
pkill -9 -f "train_ppo_ray|EngineCore|PolicyModelActor|RolloutRayActor|ray::"
sleep 5
rm -rf /tmp/ray /tmp/e2e_* /tmp/sg_*

# Verify GPUs are free (memory should be ~0 MiB)
xpu-smi stats -d 0 2>/dev/null | grep -i "mem\|util" | head -3
xpu-smi stats -d 1 2>/dev/null | grep -i "mem\|util" | head -3
```

Then re-run the failed test using the filter.

### `DeepSpeed Op Builder: Installed CUDA version X does not match...`
On NVIDIA: the container's CUDA toolkit differs from the one PyTorch was built with.

```bash
export DS_SKIP_CUDA_CHECK=1
# Then re-run
```

### `AssertionError: Torch not compiled with CUDA enabled`
A `torch.cuda.*` call is being reached on an XPU build. Confirm the PR4
device-API changes are applied — specifically `ppo_actor.py:378`:
```python
# Must be:
self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta,
    torch.accelerator.current_accelerator().type)
# NOT:
self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
```

### `PermissionError` on HuggingFace cache
Docker was previously run as root, leaving root-owned lock directories.

```bash
# Option 1: fix permissions (requires sudo)
sudo chown -R $USER ~/.cache/huggingface

# Option 2: use a fresh cache location
export HF_HOME=/tmp/hf_fresh
export HF_DATASETS_CACHE=/tmp/hf_datasets_fresh
```

### `ray.exceptions.RuntimeError: Version mismatch`
A different `ray` binary is on `PATH` than the one in your active environment.
Activate the correct environment first, or point the suites at the right one:

```bash
RAY=/path/to/env/bin/ray PYTHON=/path/to/env/bin/python \
  bash tests/test_e2e_suite_multigpu.sh
```

The suites use `ray`/`python` from `PATH` by default; `RAY=` / `PYTHON=`
override them for both the suite and its manual Ray commands.

### `grad_accum = 0` assertion
`train.batch_size` is too small for the number of actor GPUs:
```
grad_accum = train.batch_size / (micro_batch_size × actor_gpus)  must be ≥ 1
```
The suites set `train.batch_size=8` and `micro_batch_size=4` for 1-GPU actor,
giving `grad_accum=2`. If you change GPU counts manually, scale accordingly.

---

## Running on a clean machine (first time setup)

If this is the first time running on a freshly set-up machine:

```bash
# 1. Confirm the environment has the right packages
python -c "import torch, vllm, ray, deepspeed; print('all imports OK')"

# 2. Download the VLM model (multi-GPU suite only)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-2B-Instruct',
                  local_dir='/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct')
"

# 3. RL/SFT datasets are generated automatically on first suite launch.
#    To pre-generate them explicitly (optional):
python tests/prepare_e2e_data.py

# 4. Run unit tests first (quick sanity check, no GPU needed)
python -m pytest tests/ -q --ignore=tests/results

# 5. If unit tests pass, run the full suite
nohup bash tests/test_e2e_suite_multigpu.sh > tests/multigpu_run.log 2>&1 &
```
