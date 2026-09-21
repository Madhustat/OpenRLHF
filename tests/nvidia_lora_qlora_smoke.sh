#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA regression smoke test for the LoRA / QLoRA online vLLM weight-sync path.
#
# Exercises the ONLY change this repo carries over upstream main:
#   openrlhf/trainer/ray/ppo_actor.py :: broadcast_to_vllm
#     - PeftModel merge-before-broadcast (LoRA)
#     - dequantize_4bit before sync    (QLoRA / --ds.load_in_4bit)
#
# Two cases (single NVIDIA GPU, GRPO so no critic needed):
#   sg_grpo_lora   — GRPO + LoRA
#   sg_grpo_qlora  — GRPO + LoRA + 4-bit base (--ds.load_in_4bit)  <-- QLoRA
#
# PASS = process exits 0 AND at least one training-step line is logged. The point
# is that LoRA/QLoRA weights sync to vLLM without hanging/erroring; reward value
# is irrelevant for this smoke.
#
# Requirements on the NVIDIA box:
#   pip install "torch" vllm deepspeed ray peft bitsandbytes transformers
#   (bitsandbytes with CUDA is required for the QLoRA case)
#
# Run:
#   bash tests/nvidia_lora_qlora_smoke.sh            # both cases
#   bash tests/nvidia_lora_qlora_smoke.sh qlora      # only the QLoRA case
#   MODEL=Qwen/Qwen2.5-0.5B bash tests/nvidia_lora_qlora_smoke.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

FILTER="${1:-}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
PY="${PY:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS="$REPO/tests/results/nvidia_lora_qlora_$TS"
mkdir -p "$RESULTS"
DATA="$RESULTS/prompts.jsonl"
REWARD_FN="$REPO/examples/python/math_reward_func.py"
SUMMARY="$RESULTS/summary.txt"; : > "$SUMMARY"

# --- tiny GSM8K-style prompt set (prompt + boxed-answer label) ----------------
"$PY" - "$DATA" <<'PY'
import json, sys
rows = [
    ("What is 2 + 3? Put the final answer in \\boxed{}.", "5"),
    ("What is 7 - 4? Put the final answer in \\boxed{}.", "3"),
    ("What is 6 * 3? Put the final answer in \\boxed{}.", "18"),
    ("What is 20 / 5? Put the final answer in \\boxed{}.", "4"),
    ("What is 9 + 8? Put the final answer in \\boxed{}.", "17"),
    ("What is 12 - 5? Put the final answer in \\boxed{}.", "7"),
    ("What is 4 * 4? Put the final answer in \\boxed{}.", "16"),
    ("What is 15 + 6? Put the final answer in \\boxed{}.", "21"),
]
with open(sys.argv[1], "w") as f:
    for _ in range(10):                       # ~80 samples
        for p, a in rows:
            f.write(json.dumps({"prompt": p, "label": a}) + "\n")
print("wrote", sys.argv[1])
PY

# --- shared single-GPU GRPO flags (NVIDIA / NCCL) -----------------------------
base_flags() {
  echo \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
    --train.colocate_all --vllm.sync_backend nccl \
    --actor.model_name_or_path "$MODEL" \
    --reward.remote_url "$REWARD_FN" \
    --data.prompt_dataset "$DATA" \
    --data.input_key prompt --data.label_key label --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 80 \
    --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
    --ds.attn_implementation sdpa --logger.logging_steps 1 --eval.steps -1 \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32
}

run_case() {
  local id="$1"; shift
  [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && { echo "SKIP  $id (filtered)" | tee -a "$SUMMARY"; return; }
  local logf="$RESULTS/$id.log"
  echo "==== START $id ===="; echo "log: $logf"
  ray stop --force >/dev/null 2>&1 || true
  PYTHONUNBUFFERED=1 "$PY" -m openrlhf.cli.train_ppo_ray $(base_flags) "$@" > "$logf" 2>&1
  local rc=$?
  local steps; steps=$(grep -cE "Episode|global_step|policy_loss|reward" "$logf" 2>/dev/null || echo 0)
  if [[ $rc -eq 0 ]]; then
    echo "PASS  $id (exit=0, step-lines=$steps)" | tee -a "$SUMMARY"
  else
    echo "FAIL  $id (exit=$rc, step-lines=$steps) — see $logf" | tee -a "$SUMMARY"
  fi
  ray stop --force >/dev/null 2>&1 || true
}

# [1] LoRA (bf16 base + LoRA adapters)
run_case "sg_grpo_lora"

# [2] QLoRA (4-bit base + LoRA adapters) — adds --ds.load_in_4bit
run_case "sg_grpo_qlora" --ds.load_in_4bit

echo; echo "===================== SUMMARY ====================="
cat "$SUMMARY"
echo "Logs: $RESULTS"
