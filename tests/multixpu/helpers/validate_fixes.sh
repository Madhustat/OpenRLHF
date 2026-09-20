#!/usr/bin/env bash
# Re-run ONLY the cases that were fixed, one at a time, and collect the outcomes.
# Cheaper than a full 4.5 h suite re-run for checking whether the fixes took.
# In its own file so the suite's internal pkill patterns cannot match the caller.
set -uo pipefail
REPO=/home/sdp/madhu/OpenRLHF-fresh
V=/home/sdp/venvs/openrlhf-xccl-auto-detect-213
export LD_LIBRARY_PATH="$V/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export PYTHON=$V/bin/python
export RAY=$V/bin/ray
export PYTHONPATH=$REPO
export OPENRLHF_DS_TORCH_ADAM=1
cd "$REPO"

OUT=/tmp/fix_validation.txt
: > "$OUT"

CASES=(
  mg_is_tv_gating            # single delta instead of a band
  mg_ds_autotp2              # + vllm.num_engines 2
  mg_ring_attn2              # + vllm.num_engines 2
  mg_moe_experts_batched_mm  # reduced batch
  mg_sft_pretrain_mode       # flat text column
  mg_data_chat_template      # model's own template
  mg_liger                   # should now SKIP
  mg_sync_with_ray           # retried under colocate_all
  mg_sft_4bit                # NEW: isolates 4-bit from rollout
  mg_determinism             # full aligned comparison
)

for c in "${CASES[@]}"; do
    echo "=== $(date +%H:%M:%S) running $c" | tee -a "$OUT"
    bash tests/test_e2e_suite_multigpu_extended.sh "$c" > "/tmp/fix_$c.out" 2>&1
    r=$(grep -E "^(PASS|FAIL|SKIP)  $c" "/tmp/fix_$c.out" | grep -v filtered | head -1)
    echo "    ${r:-<no result line>}" | tee -a "$OUT"
done

echo | tee -a "$OUT"
echo "=== SUMMARY ===" | tee -a "$OUT"
grep -E '^    (PASS|FAIL|SKIP)' "$OUT" | sed 's/^    //' | tee -a "$OUT".final
echo "DONE" | tee -a "$OUT"
