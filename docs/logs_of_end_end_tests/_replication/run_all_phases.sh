#!/bin/bash
# =============================================================================
# OpenRLHF single-GPU XPU end-to-end validation — REPLICATION SCRIPT (Phases 0-9)
# Verified 2026-07-20 on single Intel Arc Pro B70 (BMG), 31 GB RAM.
#
# Usage:
#   bash run_all_phases.sh [phase]
#     no arg   -> run ALL phases 0..9 in order
#     0..9     -> run just that phase (e.g. `bash run_all_phases.sh 8b`)
#   valid phase tokens: 0 1 2 3 4 5 6 7 8b 8c 8 9
#
# NOTE: RL phases (8/8b/8c) use Qwen2.5-0.5B-Instruct. The 1.5B model OOMs on this
#       31 GB box under colocate_all (see ../08b_grpo/GRPO_OOM_FINDINGS.md).
#       SFT/RM/DPO (offline, single model) run fine at 1.5B; kept at 1.5B below to
#       match the original matrix — change SFT_MODEL if you want everything on 0.5B.
# =============================================================================
set -o pipefail

# ---- paths / env (EDIT THESE TWO if the layout ever changes) ----
REPO=/home/dut7054/openrlfh_exp/OpenRLHF
VENV=/home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf
ONEAPI=/opt/intel/oneapi/compiler/2025.3/bin

LOGROOT="$REPO/docs/logs_of_end_end_tests"
DATASET="/home/dut7054/openrlfh_exp/data/gsm8k_train_prompts.jsonl"
SFT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"      # offline workflows; fits at 1.5B
RL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"       # colocated RLHF; MUST be <=0.5B on 31GB box
REWARD_FUNC="examples/python/math_reward_func.py"

cd "$REPO"
source "$VENV/bin/activate"
export PATH="$ONEAPI:$PATH"          # oneAPI icpx for DeepSpeed JIT; PATH-only, do NOT source setvars.sh
RAY_BIN="$VENV/bin/ray"
TS() { date +%Y%m%d_%H%M%S; }

PHASE="${1:-all}"
run() { [ "$PHASE" = "all" ] || [ "$PHASE" = "$1" ]; }

# -----------------------------------------------------------------------------
# PHASE 0 — environment snapshot (proves XPU/Ray/vLLM/DeepSpeed are the real stack)
# -----------------------------------------------------------------------------
if run 0; then
  echo "### PHASE 0: env snapshot"
  LOG="$LOGROOT/00_env_snapshot/env_snapshot_$(TS).log"
  {
    python -c "import torch;print('torch',torch.__version__,'| xpu',torch.xpu.is_available(),'| count',torch.xpu.device_count(),'|',torch.xpu.get_device_name(0))"
    python -c "import ray,vllm,deepspeed,transformers;print('ray',ray.__version__,'| vllm',vllm.__version__,'| deepspeed',deepspeed.__version__,'| transformers',transformers.__version__)"
    python -c "import openrlhf;print('openrlhf resolves from',openrlhf.__file__)"
  } > "$LOG" 2>&1
  cat "$LOG"
fi

# -----------------------------------------------------------------------------
# PHASE 1 — CLI --help for all 4 entrypoints
# -----------------------------------------------------------------------------
if run 1; then
  echo "### PHASE 1: CLI --help"
  LOG="$LOGROOT/01_cli_help/cli_help_$(TS).log"
  : > "$LOG"
  for e in train_sft train_rm train_dpo train_ppo_ray; do
    echo "=== $e --help ===" >> "$LOG"
    timeout 60 python -m openrlhf.cli.$e --help >> "$LOG" 2>&1
    echo "$e exit=$?" | tee -a "$LOG"
  done
fi

# -----------------------------------------------------------------------------
# PHASE 2 — HF model load + generate on XPU
# -----------------------------------------------------------------------------
if run 2; then
  echo "### PHASE 2: HF model load + generate"
  LOG="$LOGROOT/02_hf_model_load/hf_model_load_$(TS).log"
  python - > "$LOG" 2>&1 <<PYEOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m="$SFT_MODEL"
tok=AutoTokenizer.from_pretrained(m)
model=AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to("xpu")
print("on device:", next(model.parameters()).device)
inp=tok("Hello", return_tensors="pt").to("xpu")
print("OUTPUT:", tok.decode(model.generate(**inp, max_new_tokens=20)[0], skip_special_tokens=True))
print("PHASE2=PASS")
PYEOF
  tail -3 "$LOG"
fi

# -----------------------------------------------------------------------------
# PHASE 3/4/5 — SFT / RM / DPO (offline, DeepSpeed, 5 optimizer steps, ckpt save)
# max_samples chosen so 5 optimizer steps fire (batch_size 2 => accum 2).
# -----------------------------------------------------------------------------
if run 3; then
  echo "### PHASE 3: SFT (5 steps)"
  B=/tmp/sft_smoke_$(TS); LOG="$LOGROOT/03_sft/sft_5steps_$(TS).log"
  python -m openrlhf.cli.train_sft --data.max_len 512 --data.dataset Open-Orca/OpenOrca \
    --data.input_key question --data.output_key response --data.max_samples 12 \
    --train.batch_size 2 --train.micro_batch_size 1 --model.model_name_or_path "$SFT_MODEL" \
    --ckpt.output_dir "$B/output" --ckpt.path "$B/checkpoints_sft" --ckpt.save_steps 5 --ckpt.save_hf \
    --logger.logging_steps 1 --eval.steps -1 --ds.zero_stage 0 --train.max_epochs 1 \
    --ds.param_dtype bf16 --ds.attn_implementation sdpa --adam.lr 5e-6 > "$LOG" 2>&1
  echo "SFT exit=$?"
fi

if run 4; then
  echo "### PHASE 4: RM (5 steps)"
  B=/tmp/rm_smoke_$(TS); LOG="$LOGROOT/04_rm/rm_5steps_$(TS).log"
  python -m openrlhf.cli.train_rm --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 12 --train.batch_size 2 --train.micro_batch_size 1 \
    --model.model_name_or_path "$SFT_MODEL" --ckpt.output_dir "$B/output" --ckpt.path "$B/checkpoints_rm" \
    --ckpt.save_steps 5 --ckpt.save_hf --logger.logging_steps 1 --eval.steps -1 --ds.zero_stage 0 \
    --train.max_epochs 1 --ds.param_dtype bf16 --ds.attn_implementation sdpa --adam.lr 5e-6 > "$LOG" 2>&1
  echo "RM exit=$?"
fi

if run 5; then
  echo "### PHASE 5: DPO (5 steps)"
  B=/tmp/dpo_smoke_$(TS); LOG="$LOGROOT/05_dpo/dpo_5steps_$(TS).log"
  python -m openrlhf.cli.train_dpo --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
    --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 12 --train.batch_size 2 --train.micro_batch_size 1 \
    --model.model_name_or_path "$SFT_MODEL" --model.beta 0.1 --ckpt.output_dir "$B/output" \
    --ckpt.path "$B/checkpoints_dpo" --ckpt.save_steps 5 --ckpt.save_hf --logger.logging_steps 1 \
    --eval.steps -1 --ds.zero_stage 0 --train.max_epochs 1 --ds.param_dtype bf16 \
    --ds.attn_implementation sdpa --adam.lr 5e-7 > "$LOG" 2>&1
  echo "DPO exit=$?"
fi

# -----------------------------------------------------------------------------
# PHASE 6 — vLLM standalone on XPU (needs __main__ guard: V1 engine spawns subprocs)
# -----------------------------------------------------------------------------
if run 6; then
  echo "### PHASE 6: vLLM standalone"
  LOG="$LOGROOT/06_vllm/vllm_generate_$(TS).log"
  VLLM_USE_V1=1 python "$LOGROOT/_replication/vllm_phase6.py" > "$LOG" 2>&1
  echo "vLLM exit=$?"; grep -E "OUTPUT:|PHASE6" "$LOG"
fi

# -----------------------------------------------------------------------------
# PHASE 7 — Ray sees XPU + worker matmul + env-var propagation
# -----------------------------------------------------------------------------
if run 7; then
  echo "### PHASE 7: Ray validation"
  LOG="$LOGROOT/07_ray/ray_worker_xpu_$(TS).log"
  "$RAY_BIN" stop --force >/dev/null 2>&1; rm -rf /tmp/ray
  python "$LOGROOT/_replication/ray_phase7.py" > "$LOG" 2>&1
  echo "Ray exit=$?"; grep -E "PHASE7|Worker result" "$LOG"
fi

# -----------------------------------------------------------------------------
# PHASE 8b / 8c / 8 — GRPO / REINFORCE++ / PPO (full Ray+vLLM+DeepSpeed loop)
# All delegate to run_ppo_phase8.sh (handles NOSET env, sleep flags, ray start).
# -----------------------------------------------------------------------------
if run 8b; then
  echo "### PHASE 8b: GRPO (group_norm)"
  LOG="$LOGROOT/08b_grpo/grpo_group_norm_0.5B_$(TS).log"; : > "$LOG"
  bash "$LOGROOT/_replication/run_ppo_phase8.sh" group_norm "$LOG" "/tmp/grpo_$(TS)"
fi

if run 8c; then
  echo "### PHASE 8c: REINFORCE++ (reinforce_baseline)"
  LOG="$LOGROOT/08c_reinforce_plus_plus/reinforce_baseline_0.5B_$(TS).log"; : > "$LOG"
  bash "$LOGROOT/_replication/run_ppo_phase8.sh" reinforce_baseline "$LOG" "/tmp/rpp_$(TS)"
fi

if run 8; then
  echo "### PHASE 8: PPO (gae + critic)"
  LOG="$LOGROOT/08_ppo/ppo_gae_0.5B_$(TS).log"; : > "$LOG"
  bash "$LOGROOT/_replication/run_ppo_phase8.sh" gae "$LOG" "/tmp/ppo_$(TS)"
fi

# -----------------------------------------------------------------------------
# PHASE 9 — checkpoint resume (train 2 epochs+save -> reload -> continue to epoch 3)
# -----------------------------------------------------------------------------
if run 9; then
  echo "### PHASE 9: checkpoint resume"
  LOG="$LOGROOT/09_checkpoint_resume/ckpt_resume_$(TS).log"; : > "$LOG"
  bash "$LOGROOT/_replication/run_ckpt_resume.sh" "$LOG"
  grep -E "Loaded the checkpoint|RUN_A_exit|RUN_B_exit" "$LOG"
fi

echo "=== done: phase(s) '$PHASE' ==="
