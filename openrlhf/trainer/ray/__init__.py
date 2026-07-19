# No implicit imports of deepspeed here to avoid vllm environment gets comtaminated
#
# create_vllm_engines/batch_vllm_engine_call require the `vllm` package, which
# is optional - PPO/GRPO training can run with `--vllm.num_engines 0`
# (DeepSpeed-only rollout, no vLLM), and vllm may not be installed at all in
# that case. Callers must import from .vllm_engine directly, inside the code
# path that actually uses vLLM (see train_ppo_ray.py's `train()`), rather than
# importing this package eagerly - an unconditional import here would block
# train_ppo_ray.py entirely even for runs that never touch vLLM. Same class of
# bug as the flash_attn import fix in ring_attn_utils.py (see
# MASTER_setup_and_fixes_guide.md Part 3.1).
