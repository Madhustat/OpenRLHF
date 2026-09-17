#!/usr/bin/env bash
# Deep-check #4 critic_update: critic optimizes in PPO; critic-freezing holds it fixed.
set -uo pipefail
REPO=/home/dut7054/madhu/experimental-e2e-baseline-1xpu
VENV=/home/dut7054/madhu/venv-torch213-xpu
export VIRTUAL_ENV=$VENV
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0 RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1 OPENRLHF_WEIGHT_PROBE=0 RAY_memory_usage_threshold=0.97
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
export OPENRLHF_DEEPCHECK_CRITIC=1
cd "$REPO"
echo "### run ppo_gae (critic should UPDATE) ###"
bash tests/test_e2e_suite_singlegpu_extended.sh sg_ppo_gae
echo "### run ppo_gae_critic_freezing (critic FROZEN first 3 steps) ###"
bash tests/test_e2e_suite_singlegpu_extended.sh sg_ppo_gae_critic_freezing
echo "### CRITIC DEEPCHECK DONE ###"
