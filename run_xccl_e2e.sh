#!/usr/bin/env bash
# In-container GRPO E2E on 2 XPUs. Backend is arg 1 (gloo control | xccl direct); default xccl.
set -uo pipefail
SYNC_BACKEND="${1:-xccl}"

export PYTHONPATH=/repo
# Ensure ninja and icpx are on system PATH for all Ray worker subprocesses.
# icpx is on the bind-mounted host oneAPI; symlink to /usr/local/bin so sh -c finds it.
ICPX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
[ -f "$ICPX" ] && ln -sf "$ICPX" /usr/local/bin/icpx 2>/dev/null || true
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:/opt/venv/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
# Let oneCCL use its verified pidfd IPC exchange (works now that the container runs with
# seccomp=unconfined + /dev/dri/by-path mounted). Debug logging so the exchange mode is visible.
# NOTE: do NOT force CCL_ZE_IPC_EXCHANGE=sockets - it overrides the working pidfd/drmfd path.
export CCL_LOG_LEVEL=debug

cd /repo
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray
sleep 3
ray start --head --num-gpus=2 2>&1 | tail -2
sleep 3

# GPU budget on this 2-XPU box: vLLM engine (1) + policy actor (1) + KL reference model (1)
# = 3 would be UNSCHEDULABLE. --train.colocate_actor_ref packs policy+reference onto ONE XPU,
# leaving vLLM on the other. The XCCL sender (policy, XPU 0) and receiver (vLLM, XPU 1) stay on
# DISTINCT physical XPUs, so this still validates the intended multi-XPU XCCL topology.
timeout 1200 python -m openrlhf.cli.train_ppo_ray \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --train.colocate_actor_ref \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.8 --vllm.enforce_eager \
    --vllm.sync_backend "$SYNC_BACKEND" \
    --actor.model_name_or_path Qwen/Qwen2.5-0.5B \
    --critic.model_name_or_path "" \
    --algo.advantage.estimator group_norm \
    --rollout.n_samples_per_prompt 4 \
    --reward.remote_url /repo/examples/python/math_reward_func.py \
    --data.prompt_dataset /repo/tests/data/gsm8k_train_prompts.jsonl \
    --data.input_key prompt --data.label_key label --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 400 \
    --train.batch_size 8 --train.micro_batch_size 4 \
    --train.max_epochs 1 \
    --ds.attn_implementation sdpa \
    --ckpt.output_dir /tmp/e2e_xccl --ckpt.save_steps -1 \
    --logger.logging_steps 1 --eval.steps -1
rc=$?
ray stop --force >/dev/null 2>&1 || true
echo "TRAIN_EXIT=$rc"
