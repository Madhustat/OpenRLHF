#!/usr/bin/env bash
# Verify GRPO trains on a single XPU across ZeRO stages 0, 1, 2.
# (Stage 3 is separate — needs the overlap_comm + FusedAdam fixes; tested elsewhere.)
# Stage 0: no adam_offload (CPU offload requires ZeRO>=1). Stages 1/2: with offload.
set -uo pipefail
REPO=/home/dut7054/madhu/experimental-e2e-baseline-1xpu
VENV=/home/dut7054/madhu/venv-torch213-xpu
MODEL=Qwen/Qwen2.5-0.5B
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0 RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1 OPENRLHF_WEIGHT_PROBE=0 RAY_memory_usage_threshold=0.97
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
OUT=$REPO/tests/results/zero_stages_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
PROMPTS=$REPO/tests/data/gsm8k_train_prompts.jsonl

run_stage() {
  local stage=$1; shift; local extra="$*"
  local log="$OUT/zero${stage}.log"
  echo "[zero$stage] starting (extra: ${extra:-none})"
  "$VENV/bin/ray" stop --force >/dev/null 2>&1; rm -rf /tmp/ray; sleep 2
  "$VENV/bin/ray" start --head --num-gpus=1 >/dev/null 2>&1; sleep 3
  "$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
    --train.colocate_all --vllm.enable_sleep --ds.enable_sleep --vllm.sync_backend gloo \
    --actor.model_name_or_path "$MODEL" \
    --reward.remote_url "$REPO/examples/python/math_reward_func.py" \
    --data.prompt_dataset "$PROMPTS" \
    --data.input_key prompt --data.label_key label --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 40 \
    --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
    --ds.attn_implementation sdpa --ds.zero_stage $stage $extra \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ckpt.output_dir /tmp/zero$stage --ckpt.save_steps -1 \
    --logger.logging_steps 1 --eval.steps -1 > "$log" 2>&1
  local rc=$?
  "$VENV/bin/ray" stop --force >/dev/null 2>&1
  local steps=$(grep -c "Global step" "$log")
  if [ $rc -eq 0 ] && [ "$steps" -ge 1 ]; then echo "PASS  ZeRO-$stage — $steps steps"
  else echo "FAIL  ZeRO-$stage — exit=$rc steps=$steps (see $log)"; fi
}

echo "==================== ZeRO STAGE SWEEP (single XPU, GRPO) ===================="
run_stage 0
run_stage 1 --ds.adam_offload
run_stage 2 --ds.adam_offload
echo "============================================================================"
echo "logs: $OUT"
