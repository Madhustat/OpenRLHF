# OpenRLHF XPU Test Suite

This document describes every test in the repository, how to run them, and what each one validates.

## Quick start

```bash
# Unit tests — no GPU required, ~40 s
python -m pytest tests/ -q

# E2E suite — Multi-GPU (2x Arc Pro B70) — 28 tests, ~2 hrs
bash tests/test_e2e_suite_multigpu.sh

# E2E suite — Single GPU — 19 tests, ~1 hr
bash tests/test_e2e_suite_singlegpu.sh

# Filter to one group or one test (works for both suites)
bash tests/test_e2e_suite_multigpu.sh grpo
bash tests/test_e2e_suite_singlegpu.sh sft
bash tests/test_e2e_suite_multigpu.sh grpo_zero2_nocolo
```

---

## Part 1 — Unit tests (`pytest tests/`)

Run with no hardware required (CPU-only), no Ray cluster, no model download.

```bash
python -m pytest tests/ -q
# Expected: 86 passed
```

### Environment setup (XPU box)

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:venv-tf/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
```

---

### `test_gloo_weight_sync.py` — 10 tests

Validates the gloo-based actor→vLLM weight-sync path introduced for Intel XPU
support (`distributed_util._GlooBroadcastCommunicator` and
`resolve_vllm_sync_backend`).

| Test | What it checks |
|---|---|
| `test_resolve_backend_auto_and_explicit` | Auto-select picks nccl on CUDA/ROCm, gloo on XPU; explicit values are honoured; impossible combos raise |
| `test_resolve_backend_explicit_nccl_on_non_nccl_raises` | `--vllm.sync_backend nccl` on a non-CUDA build raises `ValueError` |
| `test_resolve_backend_explicit_xccl_on_non_xpu_raises` | `--vllm.sync_backend xccl` on a non-XPU build raises `RuntimeError` |
| `test_resolve_backend_xccl_single_xpu_falls_back_to_gloo` | explicit xccl with 1 device falls back to gloo (oneCCL needs 1 rank per device) |
| `test_resolve_backend_unknown_raises` | Unknown backend string raises `ValueError` |
| `test_stateless_gloo_broadcast_all_ranks_receive_source` | Real 2/3/4-rank gloo broadcast — every rank receives rank-0's value |
| `test_communicator_preserves_storage_and_returns_same_tensor` | `broadcast()` updates the original tensor in-place (no storage replacement) |
| `test_communicator_broadcast_delivers_value` | CPU→gloo→CPU round-trip delivers correct value |
| `test_communicator_accepts_stream_kwarg` | `stream=` keyword accepted and ignored (PyNcclCommunicator interface parity) |
| `test_xccl_broadcast_direct_on_xpu` | Real 2-rank XCCL broadcast on XPU — needs 2 physical XPUs |

---

### `test_loss_aggregation.py` — 8 tests

Validates per-token loss aggregation and gradient normalisation logic in
`loss_utils.py`. Runs on CPU and XPU (parametrised).

| Test | What it checks |
|---|---|
| `test_token_mean_aggregation_matches_verl_dp_average` | Token-mean loss matches reference DP-average implementation |
| `test_seq_mean_token_mean_aggregation_matches_verl_dp_average` | Sequence-mean variant matches reference |
| `test_loss_batch_info_excludes_fully_masked_sequences` | Fully-masked sequences are excluded from loss denominator |
| `test_policy_loss_token_mean_aggregation_matches_full_batch` | Per-microbatch loss matches full-batch loss |
| `test_policy_kl_metric_uses_policy_ratio_when_vllm_correction_is_enabled` | KL metric uses correct ratio under IS correction |
| `test_grad_accum_global_norm_matches_global_token_mean_over_window` | Gradient-accumulation global norm matches token-mean over full window |
| `test_grad_accum_global_norm_differs_from_per_microbatch_when_uneven` | Uneven microbatch sizes produce different norms — sanity check |
| `test_policy_kl_metric_is_not_clamped` | KL metric is not clamped (regression: old code clamped incorrectly) |

---

### `test_loss_aggregation_generic.py` — 8 tests

Device-generic version of the above. Same 8 tests parametrised across `cpu`
and the active accelerator (`xpu` or `cuda`). Confirms `loss_utils.py`
device-agnostic changes produce identical results on all backends.

---

### `test_ray_env_vars.py` — 11 tests

Validates that `train_ppo_ray.py` passes user-set environment variables
through to Ray workers without overriding them with hardcoded defaults.
No Ray cluster needed — mocks `ray.init`.

| Test | What it checks |
|---|---|
| `test_nccl_debug_default` | `NCCL_DEBUG` defaults to `WARN` when unset |
| `test_ccl_log_level_default` | `CCL_LOG_LEVEL` defaults to `warn` (XPU/XCCL logging) |
| `test_tokenizers_parallelism_default` | `TOKENIZERS_PARALLELISM` defaults to `true` |
| `test_ray_zero_copy_default` | `RAY_ENABLE_ZERO_COPY_TORCH_TENSORS` defaults to `1` |
| `test_nccl_debug_user_info` | User-set `NCCL_DEBUG=INFO` is not overridden |
| `test_nccl_debug_user_trace` | User-set `NCCL_DEBUG=TRACE` is not overridden |
| `test_ccl_log_level_user_debug` | User-set `CCL_LOG_LEVEL=debug` is not overridden |
| `test_tokenizers_parallelism_user_false` | User-set `TOKENIZERS_PARALLELISM=false` is not overridden |
| `test_ray_zero_copy_user_disabled` | User-set `RAY_ENABLE_ZERO_COPY_TORCH_TENSORS=0` is not overridden |
| `test_multiple_user_overrides` | Multiple user-set vars are all preserved simultaneously |
| `test_ray_init_not_called_when_initialized` | `ray.init` is skipped when Ray is already initialized |

---

### `test_ring_attn_utils.py` — 7 tests

Validates the optional `flash_attn` fallback in `ring_attn_utils.py`.
When `flash_attn` is not installed (e.g. on Intel XPU), the module falls
back to `transformers` and `einops` equivalents.

| Test | What it checks |
|---|---|
| `test_index_first_axis_preserves_trailing_dimensions` | Fallback `index_first_axis` keeps trailing dims (regression: transformers alias flattened them) |
| `test_index_first_axis_gradient_flows_to_selected_rows` | Autograd works through the gather: selected rows get gradients |
| `test_packing_path_regression` | Drives the exact code path that failed during `--ds.packing_samples` training |
| `test_fallback_symbols_are_wired_when_flash_attn_absent` | All fallback symbols work when `flash_attn` is absent |
| `test_guard_falls_back_only_for_top_level_package` | Only `ModuleNotFoundError` for `flash_attn` activates fallback; broken installs re-raise |
| `test_all_gather_requires_initialized_process_group` | Vendored `all_gather` guards against uninitialised process group |
| `test_two_rank_all_gather_forward_and_backward` | Real cross-process all-gather: forward gathers both ranks, backward reduce-scatters |

---

### `test_vllm_device_env.py` — 8 tests

Validates `vllm_engine._configure_device_env` — the function that pins each
Ray-assigned accelerator by setting the correct device-visibility env var
before vLLM/DeepSpeed initialise.

| Test | What it checks |
|---|---|
| `test_nvidia_sets_cuda_visible_devices` | CUDA path sets `CUDA_VISIBLE_DEVICES` to the assigned GPU ID |
| `test_xpu_sets_ze_affinity_mask` | XPU path sets `ZE_AFFINITY_MASK` to the assigned XPU index |
| `test_empty_gpu_ids_raises_actionable_error` | Empty `gpu_ids` raises a clear error message |
| `test_unresolvable_control_var_raises` | Unknown device type raises `RuntimeError` |
| `test_xpu_default_ray_sets_mask_when_var_absent` | `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` → mask set correctly |
| `test_nvidia_default_ray_leaves_existing_var_untouched` | Existing `CUDA_VISIBLE_DEVICES` not overwritten by Ray default path |
| `test_ray_backend_clears_active_platform_var` | Active platform env var cleared before setting new mask (prevents compounding) |
| `test_bundle_indices_env_preserved` | Bundle index env var preserved for TP > 1 configurations |

---

### `test_distributed_backend_generic.py` — 2 tests

Integration smoke tests for the canonical distributed backend on the current
device. Spawns real subprocesses to verify two ranks can exchange tensors.

| Test | What it checks |
|---|---|
| `test_backend_is_available` | The canonical backend (gloo/xccl/nccl) reports as available on this device |
| `test_two_rank_allreduce_crosses_processes` | Two spawned ranks perform a real all-reduce and both receive the sum |

---

## Part 2 — E2E test suites

Two suites are provided depending on available hardware.

| Suite | File | Tests | Hardware | Approx time |
|---|---|---|---|---|
| Multi-GPU | `test_e2e_suite_multigpu.sh` | 28 | 2 physical GPUs | ~2 hrs |
| Single-GPU | `test_e2e_suite_singlegpu.sh` | 19 | 1 physical GPU | ~1 hr |

**Multi-GPU suite** runs the full set: actor and vLLM engine on separate GPUs
(no-colocation path), plus colocated variants, VLM, and all 9 groups.

**Single-GPU suite** runs a subset compatible with 1 GPU. All RL tests use
`colocate_all + enable_sleep` so the actor and vLLM time-share the GPU.
Supervised tests (SFT/RM/DPO) are identical in both suites.

Tests excluded from the single-GPU suite (require 2+ GPUs):
- `grpo_zero3_nocolo`, `ppo_ref_model_nocolo`, `ppo_ref_zero2_offload` — no-colocation with multiple roles
- `dapo_nocolo`, `prorlv2_nocolo` — KL reference model + vLLM needs two GPU slots
- `grpo_agent_multiturn` — large memory footprint under colocation
- `vlm_grpo` — Qwen2-VL-2B too large to fit actor + vLLM on 1 GPU
- `ppo_gae` — actor + critic + vLLM = 3 roles minimum (excluded from both suites)

### `test_e2e_suite_multigpu.sh` — Multi-GPU E2E test suite

Runs full training jobs across every major OpenRLHF algorithm and configuration.
Each test runs for 10 steps (or until the dataset is exhausted for supervised tests).

### Hardware requirements

| Configuration | GPUs needed |
|---|---|
| All RL tests (Groups 1–6) | 2 (1 actor + 1 vLLM engine) |
| All supervised tests (Groups 7–9) | 1 |
| `ppo_gae` (PPO with critic) | 3 minimum — **not included**, validated on NVIDIA |

### Environment setup

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:venv-tf/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
```

### VLM prerequisite (Group 6 only)

The VLM test uses `Qwen2-VL-2B-Instruct`. The HuggingFace model cache may be
owned by root if Docker was previously run as root. Download to a clean path first:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-2B-Instruct',
                  local_dir='/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct')
"
```

### Usage

```bash
# Run all 28 tests
bash tests/test_e2e_suite.sh

# Run one group (all tests whose ID contains the filter string)
bash tests/test_e2e_suite.sh grpo        # all GRPO variants
bash tests/test_e2e_suite.sh reinforce   # REINFORCE / REINFORCE++ / RLOO
bash tests/test_e2e_suite.sh sft         # SFT tests
bash tests/test_e2e_suite.sh dpo         # DPO / IPO / cDPO
bash tests/test_e2e_suite.sh rm          # Reward Model tests

# Run one specific test
bash tests/test_e2e_suite.sh grpo_zero2_nocolo
bash tests/test_e2e_suite.sh vlm_grpo
```

### Results

Results are written to `tests/results/e2e_suite_<timestamp>/`:
- `summary.txt` — one line per test: `PASS`, `FAIL`, or `SKIP`
- `<id>.log` — full training output for each test

Exit code is `0` if all run tests passed, `1` if any failed.

### Test catalogue

| # | Group | Test ID | Description | Steps |
|---|---|---|---|---|
| 1 | GRPO variants | `grpo_zero2_nocolo` | GRPO, ZeRO-2, no colocation — baseline path | 10 |
| 2 | GRPO variants | `grpo_zero3_nocolo` | GRPO, ZeRO-3 + adam_offload — shards model params | 10 |
| 3 | GRPO variants | `grpo_colo_sleep` | GRPO, colocate_all + sleep — actor and vLLM time-share GPUs | 10 |
| 4 | GRPO variants | `grpo_lora_nocolo` | GRPO, LoRA rank=16 — adapter-only training | 10 |
| 5 | GRPO variants | `grpo_kl_kluse` | GRPO + KL as loss (k3, use_loss) — KL added to policy loss | 10 |
| 6 | GRPO variants | `ppo_ref_model_nocolo` | GRPO + reference model KL (colocate_actor_ref) | 10 |
| 7 | GRPO variants | `ppo_ref_zero2_offload` | GRPO + reference model CPU-offloaded | 10 |
| 8 | Other RL algorithms | `reinforce_nocolo` | REINFORCE — reward × log-prob, no baseline | 10 |
| 9 | Other RL algorithms | `reinforce_baseline_nocolo` | REINFORCE++ — with per-group mean baseline | 10 |
| 10 | Other RL algorithms | `rloo_nocolo` | RLOO — leave-one-out baseline | 10 |
| 11 | Other RL algorithms | `dr_grpo_nocolo` | DR-GRPO — GRPO without std-deviation normalisation | 10 |
| 12 | Advanced RL recipes | `dapo_nocolo` | DAPO — dynamic filtering + asymmetric clip-higher | 2† |
| 13 | Advanced RL recipes | `prorlv2_nocolo` | ProRL v2 — REINFORCE++ + clip-higher + dynamic filtering | 1† |
| 14 | Advanced RL recipes | `grpo_async` | Async GRPO — generation overlaps backward pass | 10 |
| 15 | Advanced RL recipes | `grpo_agent_multiturn` | GRPO + multi-turn agent — model interacts over 1-3 steps | 10 |
| 16 | Reward shaping | `grpo_overlong_penalty` | GRPO + overlong penalty — cosine penalty past soft token limit | 10 |
| 17 | Reward shaping | `grpo_reward_norm` | GRPO + reward normalisation — standardises reward distribution | 10 |
| 18 | Reward shaping | `grpo_is_correction` | GRPO + IS correction — token-level importance sampling | 10 |
| 19 | EMA checkpoint | `grpo_ema` | GRPO + EMA — exponential moving average of policy weights | 10 |
| 20 | VLM | `vlm_grpo` | VLM GRPO — Qwen2-VL-2B, image+text, frozen vision encoder | 8 |
| 21 | SFT | `sft_zero2` | SFT, ZeRO-2, full fine-tune | 64 grad steps |
| 22 | SFT | `sft_lora` | SFT, LoRA rank=16 — adapter-only training | 64 grad steps |
| 23 | SFT | `sft_packing` | SFT + sample packing — eliminates padding waste | 64 grad steps |
| 24 | Reward Model | `rm_zero2` | Reward Model, ZeRO-2, full fine-tune | 64 grad steps |
| 25 | Reward Model | `rm_lora` | Reward Model, LoRA rank=16 | 64 grad steps |
| 26 | DPO / IPO / cDPO | `dpo_zero2` | DPO, ZeRO-2 — direct preference optimisation | 64 grad steps |
| 27 | DPO / IPO / cDPO | `dpo_ipo` | IPO — avoids DPO overoptimisation | 64 grad steps |
| 28 | DPO / IPO / cDPO | `dpo_cdpo` | cDPO — label smoothing for noisy preferences | 64 grad steps |
| — | (HW limit) | `ppo_gae` | PPO + critic (gae) — needs 3 GPUs, validated on NVIDIA | — |

† DAPO and ProRL v2 produce fewer steps because dynamic filtering discards
batches where all responses score identically (no learning signal). This is
correct algorithm behaviour, not a failure.

### Validated results (2x Arc Pro B70, torch 2.12.0+xpu)

All 28 runnable tests pass. Evidence from logs:

| Test ID | Evidence |
|---|---|
| `grpo_zero2_nocolo` | 10 steps, last_reward=0.031 |
| `grpo_zero3_nocolo` | 10 steps, last_reward=0.063 |
| `grpo_colo_sleep` | 10 steps, last_reward=0.031 |
| `grpo_lora_nocolo` | 10 steps, last_reward=0.000 |
| `grpo_kl_kluse` | 10 steps, last_reward=0.156 |
| `ppo_ref_model_nocolo` | 10 steps, last_reward=0.094 |
| `ppo_ref_zero2_offload` | 10 steps, last_reward=0.063 |
| `reinforce_nocolo` | 10 steps, last_reward=0.000 |
| `reinforce_baseline_nocolo` | 10 steps, last_reward=0.031 |
| `rloo_nocolo` | 10 steps, last_reward=0.125 |
| `dr_grpo_nocolo` | 10 steps, last_reward=0.125 |
| `dapo_nocolo` | 2 steps†, last_reward=0.344 |
| `prorlv2_nocolo` | 1 step†, last_reward=0.375 |
| `grpo_async` | 10 steps, last_reward=0.063 |
| `grpo_agent_multiturn` | 10 steps, last_reward=0.094 |
| `grpo_overlong_penalty` | 10 steps, last_reward=0.063 |
| `grpo_reward_norm` | 10 steps, last_reward=0.000 |
| `grpo_is_correction` | 10 steps, last_reward=0.125 |
| `grpo_ema` | 10 steps, last_reward=0.031 |
| `vlm_grpo` | 8 steps, last_reward=0.000 |
| `sft_zero2` | 64 grad steps, last_loss=0.628 |
| `sft_lora` | 64 grad steps, last_loss=1.030 |
| `sft_packing` | 64 grad steps, last_loss=0.627 |
| `rm_zero2` | 64 grad steps, last_loss=0.449 |
| `rm_lora` | 64 grad steps, last_loss=0.695 |
| `dpo_zero2` | 64 grad steps, last_loss=0.631 |
| `dpo_ipo` | 64 grad steps, last_loss=15.0 |
| `dpo_cdpo` | 64 grad steps, last_loss=0.615 |

---

### `test_e2e_suite_singlegpu.sh` — Single-GPU E2E test suite

All RL tests use `colocate_all + enable_sleep` (actor and vLLM time-share 1 GPU).
`gpu_memory_utilization=0.4` leaves room for the actor alongside vLLM.

```bash
bash tests/test_e2e_suite_singlegpu.sh            # all 19 tests
bash tests/test_e2e_suite_singlegpu.sh grpo        # GRPO tests only
bash tests/test_e2e_suite_singlegpu.sh sg_sft_zero2  # one test
```

Results are written to `tests/results/singlegpu_<timestamp>/`.

| # | Group | Test ID | Description | Steps |
|---|---|---|---|---|
| 1 | GRPO | `sg_grpo_baseline` | GRPO, ZeRO-2, colocate_all + sleep — single-GPU baseline | 10 |
| 2 | GRPO | `sg_grpo_lora` | GRPO, LoRA rank=16, colocate_all + sleep | 10 |
| 3 | GRPO | `sg_grpo_kl` | GRPO + KL as loss (k3), colocate_all + sleep | 10 |
| 4 | Other RL | `sg_reinforce` | REINFORCE, colocate_all + sleep | 10 |
| 5 | Other RL | `sg_reinforce_baseline` | REINFORCE++, colocate_all + sleep | 10 |
| 6 | Other RL | `sg_rloo` | RLOO, colocate_all + sleep | 10 |
| 7 | Other RL | `sg_dr_grpo` | DR-GRPO, colocate_all + sleep | 10 |
| 8 | Reward shaping | `sg_grpo_overlong_penalty` | GRPO + overlong penalty, colocate_all + sleep | 10 |
| 9 | Reward shaping | `sg_grpo_reward_norm` | GRPO + reward normalisation, colocate_all + sleep | 10 |
| 10 | Reward shaping | `sg_grpo_is_correction` | GRPO + IS correction, colocate_all + sleep | 10 |
| 11 | EMA | `sg_grpo_ema` | GRPO + EMA, colocate_all + sleep | 10 |
| 12 | Async | `sg_grpo_async` | Async GRPO, colocate_all + sleep | 10 |
| 13 | SFT | `sg_sft_zero2` | SFT, ZeRO-2, full fine-tune | 64 grad steps |
| 14 | SFT | `sg_sft_lora` | SFT, LoRA rank=16 | 64 grad steps |
| 15 | SFT | `sg_sft_packing` | SFT + sample packing | 64 grad steps |
| 16 | RM | `sg_rm_zero2` | Reward Model, ZeRO-2 | 64 grad steps |
| 17 | DPO | `sg_dpo_zero2` | DPO, ZeRO-2 | 64 grad steps |
| 18 | DPO | `sg_dpo_ipo` | IPO — avoids DPO overoptimisation | 64 grad steps |
| 19 | DPO | `sg_dpo_cdpo` | cDPO — label smoothing | 64 grad steps |

---

## Troubleshooting

**`PermissionError` on HuggingFace cache locks**
Docker was previously run as root, leaving root-owned `.locks/` directories.
Fix: set `HF_DATASETS_CACHE` and `HF_HOME` to writable paths, or run
`sudo chown -R $USER ~/.cache/huggingface`.

**`AssertionError: Torch not compiled with CUDA enabled`**
A `torch.cuda.*` call is being reached on an XPU build. Check that the PR4
device-API changes have been applied — specifically `ppo_actor.py:378`
`moving_average(..., "cuda")` must be `moving_average(..., torch.accelerator.current_accelerator().type)`.

**Ray placement group pending / test hangs**
The requested GPU count exceeds what is available. Verify `ray start --num-gpus=N`
matches the physical GPU count. On this box `N=2`. The `ppo_gae` test requires
3 GPUs and will hang on a 2-GPU box — it is excluded from the suite for this reason.

**Version mismatch: `ray` from conda shadows venv's ray**
Always use `venv-tf/bin/ray` directly. `which ray` may point to a different
version; calling bare `ray` after sourcing can cause `RuntimeError: Version mismatch`.
