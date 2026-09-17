#!/usr/bin/env bash
REPO=/home/dut7054/madhu/experimental-e2e-baseline-1xpu
VENV=/home/dut7054/madhu/venv-torch213-xpu
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0 RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1 OPENRLHF_WEIGHT_PROBE=0 RAY_memory_usage_threshold=0.97
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
export OPENRLHF_DEEPCHECK_SYNC=1
cd "$REPO"; LOG=/home/dut7054/madhu/deepcheck_sync_run.log
"$VENV/bin/ray" stop --force >/dev/null 2>&1; rm -rf /tmp/ray; sleep 2
"$VENV/bin/ray" start --head --num-gpus=1 >/dev/null 2>&1; sleep 3
"$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep --vllm.sync_backend gloo \
  --actor.model_name_or_path Qwen/Qwen2.5-0.5B \
  --reward.remote_url "$REPO/examples/python/math_reward_func.py" \
  --data.prompt_dataset "$REPO/tests/data/gsm8k_train_prompts.jsonl" \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_len 512 --data.max_samples 24 \
  --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
  --ds.attn_implementation sdpa --ds.zero_stage 2 --ds.adam_offload \
  --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
  --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
  --ckpt.output_dir /tmp/dc_sync --ckpt.save_steps -1 \
  --logger.logging_steps 1 --eval.steps -1 > "$LOG" 2>&1
"$VENV/bin/ray" stop --force >/dev/null 2>&1
echo "==================== DEEP-CHECK #2 (full-path sync) ===================="
grep -a "DEEPCHECK-SYNC-MISMATCH" "$LOG" | head
LAST=$(grep -a "DEEPCHECK-SYNC checked=" "$LOG" | tail -1)
echo "  $LAST"
if grep -qa "DEEPCHECK-SYNC-MISMATCH" "$LOG"; then echo "FAIL  #2 full-path sync — mismatch(es) found"
elif [ -n "$LAST" ]; then echo "PASS  #2 full-path sync — vLLM params bit-match broadcast values"
else echo "INCONCLUSIVE #2 — no comparable params logged (see $LOG)"; fi
echo "SYNC DONE"
