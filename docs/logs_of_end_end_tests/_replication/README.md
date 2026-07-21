# Replication scripts — OpenRLHF single-GPU XPU validation (Phases 0-9)

All commands used in the 2026-07-20 validation run, saved as runnable scripts.
Verified on a single Intel Arc Pro B70 (BMG), 31 GB RAM.

## Files
- **`run_all_phases.sh`** — master driver. Run all phases or one at a time.
- `run_ppo_phase8.sh` — RL loop (GRPO/REINFORCE++/PPO), parameterized by estimator. Handles the
  `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` env, sleep flags, and `ray start`.
- `run_ckpt_resume.sh` — Phase 9 (train→save→reload→continue).
- `vllm_phase6.py` — Phase 6 vLLM standalone (has the required `__main__` guard).
- `ray_phase7.py` — Phase 7 Ray-sees-XPU worker test.

## How to run
```bash
cd /home/dut7054/openrlfh_exp/OpenRLHF
# everything, phases 0 -> 9 in order:
bash docs/logs_of_end_end_tests/_replication/run_all_phases.sh
# or a single phase (tokens: 0 1 2 3 4 5 6 7 8b 8c 8 9):
bash docs/logs_of_end_end_tests/_replication/run_all_phases.sh 8b
```
Each phase writes a timestamped `.log` into its numbered folder under `../` (e.g. `../03_sft/`).

## Prerequisites (already true on this machine — edit the two vars at the top of
`run_all_phases.sh` if the layout changes)
- venv: `/home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf` (torch+xpu, ray,
  vllm-XPU, deepspeed). Used for the Python ENV only.
- Code runs from the clone at `/home/dut7054/openrlfh_exp/OpenRLHF` (cwd shadows the editable install).
- oneAPI compiler on PATH via `/opt/intel/oneapi/compiler/2025.3/bin` (PATH-only; never source setvars.sh).
- Dataset `data/gsm8k_train_prompts.jsonl` present (used by RL phases).

## Model choice (IMPORTANT on the 31 GB box)
- SFT/RM/DPO: Qwen2.5-**1.5B**-Instruct is fine (offline, single model).
- PPO/GRPO/REINFORCE++: Qwen2.5-**0.5B**-Instruct — 1.5B OOMs under colocate_all on 31 GB RAM.
  See `../08b_grpo/GRPO_OOM_FINDINGS.md`. On a bigger-RAM / 2-GPU box, 1.5B works (teammate runbook).

## What each phase proves — see `../` folders and the main summary. All 12 passed 2026-07-20.
