#!/usr/bin/env bash
# =============================================================================
# Tier 2 — deep-check E2E (instrumented real run). Covers the run-based angles:
#   #5 loss_finite    — every step's metrics finite (OPENRLHF_DEEPCHECK_FINITE hook)
#   #3 weight_update  — trained/exported weights differ from base (real optimization)
#   #6 memory_stability — XPU device memory does not steadily grow over the run
# Self-contained angles (#1,#2,#8,#9,#10) are in tests/test_deepcheck_xpu.py.
# =============================================================================
set -uo pipefail

REPO=/home/dut7054/madhu/experimental-e2e-baseline-1xpu
VENV=/home/dut7054/madhu/venv-torch213-xpu
MODEL=Qwen/Qwen2.5-0.5B
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1 OPENRLHF_WEIGHT_PROBE=0 RAY_memory_usage_threshold=0.97
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
export OPENRLHF_DEEPCHECK_FINITE=1          # enable per-step finite scan (#5)

OUT=$REPO/tests/results/deepcheck_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
LOG=$OUT/run.log; MEM=$OUT/xpu_mem.csv; HF=/tmp/deepcheck_hf

"$VENV/bin/ray" stop --force >/dev/null 2>&1; sleep 2; rm -rf /tmp/ray "$HF"

# --- #6: sample XPU device memory in the background every 5s -----------------
( while true; do
    used=$(xpu-smi stats -d 0 2>/dev/null | grep -i "GPU Memory Used" | grep -oE "current: [0-9]+" | grep -oE "[0-9]+")
    echo "$(date +%s),${used:-NA}" >> "$MEM"; sleep 5
  done ) & MEM_PID=$!

echo "[deepcheck] starting instrumented GRPO run (finite scan on, save_hf on)..." | tee "$LOG"
"$VENV/bin/ray" start --head --num-gpus=1 >/dev/null 2>&1; sleep 3
"$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
    --train.colocate_all --vllm.enable_sleep --ds.enable_sleep --vllm.sync_backend gloo \
    --actor.model_name_or_path "$MODEL" \
    --reward.remote_url "$REPO/examples/python/math_reward_func.py" \
    --data.prompt_dataset "$REPO/tests/data/gsm8k_train_prompts.jsonl" \
    --data.input_key prompt --data.label_key label --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 160 \
    --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
    --ds.attn_implementation sdpa --ds.zero_stage 2 --ds.adam_offload \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ckpt.output_dir "$HF" --ckpt.save_steps 999 --ckpt.save_hf \
    --logger.logging_steps 1 --eval.steps -1 >> "$LOG" 2>&1
RC=$?
"$VENV/bin/ray" stop --force >/dev/null 2>&1
kill "$MEM_PID" 2>/dev/null

echo; echo "==================== DEEP-CHECK RESULTS ===================="

# #5 finite
if grep -q "DEEPCHECK-FINITE-VIOLATION" "$LOG"; then echo "FAIL  #5 loss_finite — NaN/Inf detected"
elif grep -q "DEEPCHECK-FINITE OK" "$LOG"; then echo "PASS  #5 loss_finite — all per-step metrics finite ($(grep -c 'DEEPCHECK-FINITE OK' "$LOG") steps)"
else echo "FAIL  #5 loss_finite — run did not reach any step (rc=$RC)"; fi

# #3 weight_update
HFDIR=$(dirname "$(find "$HF" -name config.json 2>/dev/null | head -1)")
if [ -n "$HFDIR" ] && [ -f "$HFDIR/config.json" ]; then
    "$VENV/bin/python" "$REPO/tests/deepcheck_weight_update.py" "$MODEL" "$HFDIR" 2>&1 | tee -a "$LOG" | grep -E "DEEPCHECK-WEIGHTUPDATE"
    grep -q "DEEPCHECK-WEIGHTUPDATE OK" "$LOG" && echo "PASS  #3 weight_update" || echo "FAIL  #3 weight_update"
else
    echo "FAIL  #3 weight_update — no exported HF model found under $HF"
fi

# #6 memory_stability — compare first-third vs last-third mean of XPU mem
"$VENV/bin/python" - "$MEM" <<'PY'
import sys
rows=[]
for ln in open(sys.argv[1]):
    try:
        _,v=ln.strip().split(",");
        if v!="NA": rows.append(int(v))
    except Exception: pass
if len(rows)<6:
    print("SKIP  #6 memory_stability — too few samples (%d)"%len(rows)); sys.exit()
n=len(rows)//3
first=sum(rows[:n])/n; last=sum(rows[-n:])/n
growth=(last-first)/max(first,1)*100
print(f"  XPU mem first-third avg={first} MiB, last-third avg={last} MiB, growth={growth:.1f}%")
print("PASS  #6 memory_stability — no steady growth" if growth < 15 else
      "WARN  #6 memory_stability — mem grew %.1f%% (investigate)"%growth)
PY
echo "============================================================"
echo "logs: $LOG"; echo "mem:  $MEM"
