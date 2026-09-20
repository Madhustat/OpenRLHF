#!/bin/bash
# Usage: run_ppo_phase8.sh <estimator> <logfile> <ckpt_dir>
# estimator: group_norm | reinforce_baseline | gae
# Based on the user's PROVEN single-GPU config: colocate_all + enable_sleep pair + gpu_mem 0.4
set -o pipefail
ESTIMATOR="$1"
LOGFILE="$2"
CKPT_DIR="$3"

cd /home/dut7054/openrlfh_exp/OpenRLHF
source /home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf/bin/activate
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
REWARD_FUNC="examples/python/math_reward_func.py"
DATASET="/home/dut7054/openrlfh_exp/data/gsm8k_train_prompts.jsonl"

# Ray must start externally with the ACTUAL detected XPU count (Ray does not auto-detect Intel XPU).
# NOSET: tell Ray NOT to mask the XPU device inside the worker (too late on XPU) -- OpenRLHF pins each
# role to its Ray-assigned GPU id explicitly. Required per MULTIGPU_LORA_ZERO3_fixes_and_flags.md.
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
# This box has 31GB host RAM; peak transient use during experience-making + optimizer brushes the
# default 0.95 kill threshold. Raise it (swap covers a brief spike) so Ray doesn't preemptively kill.
export RAY_memory_usage_threshold=0.95
RAY_BIN=/home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf/bin/ray
"$RAY_BIN" stop --force >/dev/null 2>&1
rm -rf /tmp/ray >/dev/null 2>&1
NUM_GPUS=$(python -c "import torch; print(torch.xpu.device_count())")
echo "Detected NUM_GPUS=$NUM_GPUS" | tee -a "$LOGFILE"
# Cap Ray's object store (defaults to ~30% of RAM = ~9GB here; this workload uses <1MB of it).
# Freeing that reservation gives the colocated actor/vLLM/ref the host-RAM headroom they need.
"$RAY_BIN" start --head --num-gpus="$NUM_GPUS" --object-store-memory=2000000000 >>"$LOGFILE" 2>&1

# gae needs a critic model (a 4th role); other estimators (group_norm/reinforce_baseline) do not.
CRITIC_ARG=""
if [ "$ESTIMATOR" = "gae" ]; then
  CRITIC_ARG="--critic.model_name_or_path $MODEL --critic.num_nodes 1 --critic.num_gpus_per_node 1"
fi

# max_samples 20 / rollout.batch_size 4 = 5 rollout steps
CMD="python -m openrlhf.cli.train_ppo_ray \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 $CRITIC_ARG \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 --vllm.sync_backend gloo --vllm.enforce_eager \
  --train.colocate_all --train.colocate_actor_ref \
  --vllm.enable_sleep --ds.enable_sleep \
  --algo.advantage.estimator $ESTIMATOR --algo.kl.init_coef 0 \
  --rollout.n_samples_per_prompt 2 --rollout.batch_size 4 \
  --actor.model_name_or_path $MODEL \
  --reward.remote_url $REWARD_FUNC \
  --data.prompt_dataset $DATASET \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_samples 20 --data.max_len 384 \
  --train.batch_size 4 --train.micro_batch_size 1 \
  --ds.attn_implementation sdpa --ds.zero_stage 2 \
  --vllm.gpu_memory_utilization 0.4 \
  --train.num_episodes 1 --train.max_epochs 1 \
  --rollout.max_new_tokens 200 \
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1 \
  --ckpt.output_dir $CKPT_DIR"

echo "COMMAND: $CMD" | tee -a "$LOGFILE"
timeout 1500 $CMD >>"$LOGFILE" 2>&1
EXIT_CODE=$?
echo "exit_code=$EXIT_CODE" | tee -a "$LOGFILE"
"$RAY_BIN" stop --force >/dev/null 2>&1
exit $EXIT_CODE
