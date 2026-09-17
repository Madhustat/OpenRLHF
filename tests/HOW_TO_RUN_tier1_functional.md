# Tier 1 — Functional / breadth suite (single-GPU XPU)

**Purpose:** prove every OpenRLHF workflow *runs* end-to-end on one Intel XPU and
produces plausible output (steps / loss / eval). This is the coverage map, not the
correctness gate — for invariant checks (weights actually update, exact sync, no
leaks, resume equivalence) see **Tier 2**: `HOW_TO_RUN_tier2_deepcheck.md`.

49 cases total: 18 (baseline) + 31 (extended). Includes single-XPU `sg_dapo` and
`sg_prorlv2` (upstream examples/scripts parity for DAPO / ProRL-v2).

---

## Stack (this box: dut7054, torch-2.13 XPU)

| Component | Version |
|---|---|
| Python | 3.13.14 (uv venv) |
| torch | 2.13.0+xpu |
| triton-xpu | 3.7.2 |
| vLLM | 0.27.1+xpu (source build @ `6e448d0ea`) |
| vllm-xpu-kernels | 0.1.12 |
| transformers | 5.16.0 |
| kernels | 0.16.2 |
| tokenizers | 0.23.2 |
| torchvision | 0.28.0+cpu (vLLM import dep; not for training) |
| deepspeed / ray / mpi4py | 0.19.1 / 2.55.0 / 4.1.2 |
| venv | `/home/dut7054/madhu/venv-torch213-xpu` |

---

## Environment (export before running)

```bash
export VIRTUAL_ENV=/home/dut7054/madhu/venv-torch213-xpu
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"   # venv lib FIRST (libsycl)
export PYTHONPATH="/home/dut7054/madhu/experimental-e2e-baseline-1xpu:$PYTHONPATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0                   # pin to one XPU
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1       # else actor+vLLM collide
export OPENRLHF_DS_TORCH_ADAM=1                              # no icpx -> torch AdamW
export OPENRLHF_WEIGHT_PROBE=0
export RAY_memory_usage_threshold=0.97                       # 31GB box headroom
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
```

Datasets auto-generate on first run via `tests/prepare_e2e_data.py` (GSM8K prompts +
SFT parquet). RM/DPO preference data and models auto-download from HF.

---

## How to run

```bash
cd /home/dut7054/madhu/experimental-e2e-baseline-1xpu

# everything (both suites, one combined summary) — several hours
bash tests/run_all_singlegpu.sh

# just the 18 baseline cases
bash tests/test_e2e_suite_singlegpu_torch213.sh

# just the 29 extended cases
bash tests/test_e2e_suite_singlegpu_extended.sh

# filter to one case or group (substring match)
bash tests/test_e2e_suite_singlegpu_extended.sh ckpt        # all ckpt_* cases
bash tests/test_e2e_suite_singlegpu_torch213.sh sg_grpo_kl  # one case
```

Results: `tests/results/singlegpu_<ts>/summary.txt` (baseline),
`tests/results/singlegpu_extended_<ts>/summary.txt` (extended), and
`tests/results/all_singlegpu_<ts>/combined_summary.txt` (combined).

Each case cleans orphans + `rm /tmp/ray` + logs `[HEALTH] XPU OK` before starting.

---

## Latest result: 47 PASS / 2 hardware-bound (of 49)

**Sample-packing now works** — upgrading to **transformers 5.16.0** removed the
flash-attn2/torch-2.13 kernel blocker. The 5 packing/dynamic-batch cases all pass:
- `sg_sft_packing`, `sg_rm_packing`, `sg_dpo_packing` — pass outright on 5.16.
- `sg_grpo_packing` — pass with `adam_offload` dropped (host-RAM fit).
- `sg_grpo_dynamic_batch` — pass with reduced-memory settings
  (`max_tokens_per_gpu 8192`, `n_samples 2`, `max_len 384`, `ASYNC_NUM_TASKS=4`).

The **2 remaining non-passes are hardware-bound, not code defects:**
- **`sg_ppo_gae_no_sleep`:** XPU VRAM OOM — 4 models resident, both sleeps off; needs
  sleep or >1 GPU.
- **`sg_grpo_vlm` at Qwen2-VL-2B:** OOM on one XPU. The VLM *pipeline* works on XPU
  (validated with SmolVLM-256M, `--data.max_len 2048`, `--train.micro_batch_size 1`).

> Note: the two RL packing/dynamic-batch passes use documented **reduced-memory
> settings** to fit the single-XPU 31 GB host-RAM ceiling — the feature works, at
> reduced scale; full-scale needs more RAM / multiple GPUs.

See `MODIFICATIONS_this_box.md` for every per-case setting deviation.

---

## Local adaptations baked into these suites (vs upstream)

- Device-agnostic EMA (`ppo_actor.py`), ZeRO-3 `overlap_comm` + FusedAdam-sleep
  guards (`deepspeed_utils.py`), opt-in `torch_adam` (`deepspeed.py`).
- Resume-aware and eval-on-load harness checks (see `MODIFICATIONS_this_box.md`).
- `adam_offload` dropped on 6 host-RAM-bound cases (optimizer kept on idle VRAM).
