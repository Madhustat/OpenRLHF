# Modifications on this box (dut7054, torch-2.13 XPU) — traceability record

Every deviation from the as-posted suites / upstream, so a PASS obtained under changed
settings is never mistaken for a default-config PASS. Last updated 2026-09-17.

Stack: Python 3.13.14, torch 2.13.0+xpu, vLLM 0.27.1+xpu (src @6e448d0ea),
vllm-xpu-kernels 0.1.12, **transformers 5.16.0**, **kernels 0.16.2**, **tokenizers 0.23.2**,
torchvision 0.28.0+cpu, deepspeed 0.19.1, ray 2.55.0, mpi4py 4.1.2.
venv: `/home/dut7054/madhu/venv-torch213-xpu`.

## A. Per-case setting changes (affect how a PASS was obtained)

All are **memory/scale fits for the single-XPU 31 GB host-RAM ceiling** — not changes to the
algorithm/feature under test. Each carries an inline comment at its case definition.

| Case | File:where | Change | Why | Result |
|---|---|---|---|---|
| `sg_reinforce_baseline` | torch213 | drop `--ds.adam_offload` | host-RAM OOM | PASS |
| `sg_grpo_reward_norm` | torch213 | drop `--ds.adam_offload` | host-RAM OOM | PASS |
| `sg_grpo_kl` | torch213 | drop `--ds.adam_offload` | host-RAM OOM (ref-model role) | PASS |
| `sg_ppo_gae` | extended | drop `--ds.adam_offload` | host-RAM OOM (actor+critic) | PASS |
| `sg_ppo_gae_critic_freezing` | extended | drop `--ds.adam_offload` | host-RAM OOM (actor+critic) | PASS |
| `sg_grpo_reward_offload` | extended | GRPO_BASE minus `adam_offload` | host-RAM OOM (reward on host) | PASS |
| `sg_grpo_packing` | extended | drop `--ds.adam_offload` | host-RAM OOM | PASS (10 steps) |
| `sg_grpo_dynamic_batch` | extended | `max_tokens_per_gpu 8192`, `n_samples 4→2`, `max_len 384`, `max_samples 40`, `ASYNC_NUM_TASKS=4` (keeps `adam_offload`) | host-RAM OOM; full-size dynamic batching needs more RAM/GPUs | PASS (5 steps) |
| `sg_grpo_vlm` | extended | `max_len 2048`, `micro_batch_size 1` (+ `VLM_MODEL=SmolVLM-256M` override) | 2B OOMs on 1 XPU; image tokens need larger max_len; micro_batch 1 avoids multimodal collate mismatch | PASS at 256M (pipeline; reward=0 due to reward-fn/dataset mismatch) |

NOT modified (verified): `sg_grpo_no_adam_offload`, `sg_grpo_no_sleep`,
`sg_grpo_vllm_sleep_only`, `sg_grpo_ds_sleep_only`. `adam_offload` was only ever *dropped*,
never *added*. `sg_ppo_gae_no_sleep` fix (if pursued) is lower `gpu_memory_utilization` /
`micro_batch_size`, NOT adding offload.

## B. Code changes (openrlhf)

- `trainer/ray/ppo_actor.py`: EMA `moving_average(..., "cuda")` -> `torch.accelerator.current_accelerator().type` (device-agnostic; fixes `sg_grpo_ema`). **Belongs in the device-agnostic PR.**
- `utils/deepspeed/deepspeed_utils.py`: ZeRO-3 `overlap_comm=False` explicit (1a) + FusedAdam-capability guard for DS-sleep offload (1b). *(Note: single-XPU ZeRO-3 was found to run even without these on DS 0.19.1 + adam_offload; they remain needed for multi-XPU / adam_offload-off.)*
- `utils/deepspeed/deepspeed.py`: opt-in `torch_adam` via `OPENRLHF_DS_TORCH_ADAM=1` (2).

## C. New test cases added (extended suite)

- `sg_grpo_ckpt_load_eval` — load checkpoint, continue training + in-training eval (loaded weights used across steps). PASS.
- `sg_grpo_ckpt_eval_only` — pure load + evaluate (eval-on-load, 0 training steps). PASS.
- `sg_dapo` — DAPO (GRPO + dynamic filtering + clip-higher `eps_clip_low_high 0.2 0.28` + KL-loss) on single XPU. PASS (9 steps). Needs `--ref.num_nodes/num_gpus_per_node 1` (colocated KL ref) + adam_offload dropped (host-RAM). Closes upstream `train_dapo_ray_hybrid_engine.sh` parity.
- `sg_prorlv2` — ProRL-v2 (REINFORCE++ + dynamic filtering + clip-higher + KL-loss) on single XPU. PASS (8 steps). Same ref-colocation + adam_offload notes. Closes upstream `train_prorlv2_math_hybrid_engine.sh` parity.
  - Both: dynamic filtering discards no-reward-variance batches, so step count may be < 10 — expected behaviour, not a failure.

## D. Harness / trainer instrumentation (all opt-in, default-off)

- `OPENRLHF_EVAL_ON_LOAD=1` (`ppo_trainer.py`): evaluate the restored checkpoint before training, then stop — powers `sg_grpo_ckpt_eval_only`.
- Resume-aware + eval-aware pass checks in the extended suite helpers (`*ckpt_resume*`, `*ckpt_eval_only*`).
- Per-case cleanup + `[HEALTH] XPU OK` probe in both suites' `ray_start`.
- Deep-check hooks (Tier 2): `OPENRLHF_DEEPCHECK_FINITE`, `OPENRLHF_DEEPCHECK_CRITIC`, `OPENRLHF_DEEPCHECK_SYNC`.

## E. Environment / dependency changes (not PRs)

- **transformers 5.7.0 → 5.15.0 → 5.16.0** — 5.16 removed the packing flash-attn2/torch-2.13 kernel blocker (the 3 supervised packing cases pass outright on 5.16).
- **kernels → 0.16.2**, **tokenizers → 0.23.2** (required by transformers 5.16).
- **torchvision 0.28.0+cpu** (vLLM import dep), **mpi4py 4.1.2** (supervised trainers).
- `tests/prepare_e2e_data.py` (new) generates the RL prompts + SFT parquet.
- `run_all_singlegpu.sh` exports the full 2.13 env and calls the torch213 baseline suite.

## Known non-passes (root-caused, not code defects)

- `sg_ppo_gae_no_sleep` — XPU VRAM OOM (4 models resident, both sleeps off); needs sleep or >1 GPU.
- `sg_grpo_vlm` at Qwen2-VL-2B — OOM on 1 XPU (works at SmolVLM-256M; see row above).
