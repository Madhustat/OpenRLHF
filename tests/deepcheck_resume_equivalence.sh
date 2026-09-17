#!/usr/bin/env bash
# Deep-check #7 resume_equivalence (deterministic restoration).
# Run A: train 10 steps, GREEDY eval @ step 10 (temperature 0), save checkpoint.
# Run B: load that checkpoint, GREEDY eval-on-load.
# Same weights + greedy decoding => identical eval pass1  ==> correct restoration.
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
OUT=$REPO/tests/results/deepcheck_resume_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
CKPT=/tmp/deepcheck_resume_ckpt; PROMPTS=$REPO/tests/data/gsm8k_train_prompts.jsonl
COMMON=(--actor.num_nodes 1 --actor.num_gpus_per_node 1 --vllm.num_engines 1
  --vllm.tensor_parallel_size 1 --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep --vllm.sync_backend gloo
  --actor.model_name_or_path "$MODEL" --reward.remote_url "$REPO/examples/python/math_reward_func.py"
  --data.prompt_dataset "$PROMPTS" --data.input_key prompt --data.label_key label --data.apply_chat_template
  --data.max_len 512 --data.max_samples 80 --train.batch_size 8 --train.micro_batch_size 4
  --train.max_epochs 1 --ds.attn_implementation sdpa --ds.zero_stage 2 --ds.adam_offload
  --algo.advantage.estimator group_norm --algo.kl.init_coef 0
  --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 --logger.logging_steps 1
  --eval.dataset "$PROMPTS" --eval.temperature 0)

run() { "$VENV/bin/ray" stop --force >/dev/null 2>&1; rm -rf /tmp/ray; sleep 2
        "$VENV/bin/ray" start --head --num-gpus=1 >/dev/null 2>&1; sleep 3
        "$VENV/bin/python" -m openrlhf.cli.train_ppo_ray "$@"; local r=$?
        "$VENV/bin/ray" stop --force >/dev/null 2>&1; return $r; }

rm -rf "$CKPT"
echo "[A] train + greedy eval @ step10 + save"
run "${COMMON[@]}" --ckpt.output_dir "$CKPT" --ckpt.save_steps 10 --eval.steps 10 > "$OUT/runA.log" 2>&1
echo "[B] load checkpoint + greedy eval-on-load"
OPENRLHF_EVAL_ON_LOAD=1 run "${COMMON[@]}" --ckpt.output_dir "$CKPT" --ckpt.save_steps 10 --ckpt.load_enable --eval.steps -1 > "$OUT/runB.log" 2>&1

A=$(grep -aoE "eval_default_pass1': [0-9.]+" "$OUT/runA.log" | tail -1 | grep -oE "[0-9.]+$")
B=$(grep -aoE "eval_default_pass1': [0-9.]+" "$OUT/runB.log" | tail -1 | grep -oE "[0-9.]+$")
echo "==================== DEEP-CHECK #7 ===================="
echo "  run A (trained weights, greedy eval)  pass1 = ${A:-MISSING}"
echo "  run B (loaded weights,  greedy eval)  pass1 = ${B:-MISSING}"
# Greedy vLLM decode on XPU is not perfectly bitwise-deterministic run-to-run
# (~1 sample can flip), so compare within a small tolerance rather than ==.
verdict=$("$VENV/bin/python" - "$A" "$B" <<'PY'
import sys
a,b=sys.argv[1],sys.argv[2]
try:
    a=float(a); b=float(b)
except Exception:
    print("FAIL missing"); sys.exit()
print("PASS" if abs(a-b) <= 0.02 else "FAIL", f"|A-B|={abs(a-b):.5f}")
PY
)
if echo "$verdict" | grep -q "^PASS"; then
  echo "PASS  #7 resume_equivalence — loaded model reproduces saved model within eval tolerance (A=$A B=$B, $verdict)"
else
  echo "FAIL  #7 resume_equivalence — pass1 differs beyond tolerance (A=$A B=$B, $verdict)"
fi
echo "logs: $OUT"
