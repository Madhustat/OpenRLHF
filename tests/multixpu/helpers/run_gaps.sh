#!/usr/bin/env bash
# Run ONLY the "yet to test" cases, not the whole 28-case base suite, so this fits
# inside an hour instead of 2.5 h.
#
# Each invocation is wrapped in `timeout` from the OUTSIDE, because the base e2e
# suite has no per-case budget of its own -- that is exactly what cost 2h46m earlier
# when one case hung. 900 s is ~3x a healthy RL case (~270 s measured).
#
# Own file so the suites' internal `pkill -f` patterns cannot match the caller's
# command line and kill it mid-run.
set -uo pipefail
REPO=/home/sdp/madhu/OpenRLHF-fresh
V=/home/sdp/venvs/openrlhf-xccl-auto-detect-213
export LD_LIBRARY_PATH="$V/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export PYTHON=$V/bin/python
export RAY=$V/bin/ray
export PYTHONPATH=$REPO
export OPENRLHF_DS_TORCH_ADAM=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1}
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
cd "$REPO"

OUT=/tmp/gap_results.txt
: > "$OUT"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }

# suite | case | what it closes
CASES=(
  "base|rloo_nocolo|RLOO leave-one-out advantage"
  "base|dr_grpo_nocolo|Dr. GRPO (no std-dev normalization)"
  "base|grpo_ema|EMA moving-average of policy weights"
  "base|grpo_overlong_penalty|Overlong reward penalty"
  "base|grpo_reward_norm|Reward normalization path"
  "base|ppo_ref_model_nocolo|colocate_actor_ref (partial colocation)"
  "base|dpo_ipo|DPO IPO loss variant"
  "base|dpo_cdpo|DPO cDPO label smoothing"
  "ext|mg_determinism_sft|Determinism on SFT (localises the RL determinism finding)"
  "ext|mg_moe_aux_loss|MoE router aux loss, re-pointed at tiny Qwen2Moe"
  "ext|mg_moe_experts_batched_mm|MoE batched_mm, re-pointed at tiny MoE"
  "ext|mg_data_chat_template|Custom chat template with more samples"
)

say "=== running ${#CASES[@]} gap cases ==="
for entry in "${CASES[@]}"; do
    IFS='|' read -r suite cid what <<< "$entry"
    if [[ "$suite" == "base" ]]; then
        script=tests/test_e2e_suite_multigpu.sh
    else
        script=tests/test_e2e_suite_multigpu_extended.sh
    fi
    say "START $cid — $what"
    timeout --kill-after=60 900 bash "$script" "$cid" > "/tmp/gap_$cid.out" 2>&1
    rc=$?
    line=$(grep -E "^(PASS|FAIL|SKIP)  ?$cid" "/tmp/gap_$cid.out" 2>/dev/null | grep -v filtered | head -1)
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        say "  TIMEOUT $cid (killed at 900s)"
    elif [[ -n "$line" ]]; then
        say "  $line"
    else
        why=$(grep -ohE "(AssertionError|RuntimeError|ValueError|AttributeError|TypeError|ImportError|ModuleNotFoundError|OutOfMemoryError)[^\"]{0,120}" "/tmp/gap_$cid.out" 2>/dev/null | tail -1)
        say "  NO RESULT LINE $cid rc=$rc ${why:+| $why}"
    fi
done

say "=== SUMMARY ==="
grep -E '^\[.*\]   (PASS|FAIL|SKIP|TIMEOUT|NO RESULT)' "$OUT" | sed 's/^\[[^]]*\]  //' | tee -a "$OUT"
say "DONE"
