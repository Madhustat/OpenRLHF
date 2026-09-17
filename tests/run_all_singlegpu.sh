#!/usr/bin/env bash
# =============================================================================
# Run ALL single-GPU E2E cases (44 total) in one go.
# =============================================================================
#
#   Suite 1  tests/test_e2e_suite_singlegpu.sh            cases  1-18  (baseline)
#   Suite 2  tests/test_e2e_suite_singlegpu_extended.sh   cases 19-44  (gap coverage)
#
# Both suites use identical helpers, base config and PASS criteria, so their
# results are directly comparable. This wrapper runs them back to back and
# prints one combined summary.
#
# Usage:
#   bash tests/run_all_singlegpu.sh                 # all 44
#   bash tests/run_all_singlegpu.sh sg_ppo_gae      # filter, applied to both suites
#
# Exit code: 0 only if BOTH suites had zero failures.
#
# Expect several hours. Each RL case starts and stops its own Ray head, so the
# suites are safe to interrupt between cases (Ctrl-C, then `ray stop --force`).
# =============================================================================
set -uo pipefail

# --- torch 2.13 XPU environment (this box) ----------------------------------
# The extended suite reads PYTHON/RAY from PATH and relies on the caller for the
# venv + env vars; export them here so BOTH suites run on the torch-2.13 stack.
export VIRTUAL_ENV=/home/dut7054/madhu/venv-torch213-xpu
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$VIRTUAL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1          # no icpx JIT path -> torch-native AdamW
export OPENRLHF_WEIGHT_PROBE=0
export RAY_memory_usage_threshold=0.97   # 31GB box: allow a little more headroom
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite
# ----------------------------------------------------------------------------

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# openrlhf is NOT pip-installed in the venv (only vllm is editable); the extended
# suite doesn't set PYTHONPATH itself, so resolve imports to THIS checkout here.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
FILTER=${1:-}
TS=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/tests/results/all_singlegpu_$TS
mkdir -p "$OUT"
COMBINED=$OUT/combined_summary.txt

echo "══════════════════════════════════════════════════════════════"
echo " OpenRLHF single-GPU: ALL 44 cases"
echo " results -> $OUT"
[[ -n "$FILTER" ]] && echo " filter  -> $FILTER"
echo "══════════════════════════════════════════════════════════════"

rc1=0; rc2=0

echo
echo ">>> SUITE 1/2 — baseline, cases 1-18"
bash "$REPO/tests/test_e2e_suite_singlegpu_torch213.sh" $FILTER 2>&1 \
    | tee "$OUT/suite1_baseline.log" || rc1=$?

echo
echo ">>> SUITE 2/2 — extended, cases 19-44"
bash "$REPO/tests/test_e2e_suite_singlegpu_extended.sh" $FILTER 2>&1 \
    | tee "$OUT/suite2_extended.log" || rc2=$?

# ---- merge the two per-suite summaries -------------------------------------
# Each suite writes tests/results/<name>_<timestamp>/summary.txt. Pick the most
# recent of each rather than guessing the timestamp.
s1=$(ls -1dt "$REPO"/tests/results/singlegpu_2* 2>/dev/null | head -1)
s2=$(ls -1dt "$REPO"/tests/results/singlegpu_extended_* 2>/dev/null | head -1)

{
    echo "OpenRLHF single-GPU — combined summary ($TS)"
    echo
    echo "── Suite 1: baseline (cases 1-18) ───────────────────────────"
    [[ -f "$s1/summary.txt" ]] && cat "$s1/summary.txt" || echo "(summary not found: $s1)"
    echo
    echo "── Suite 2: extended (cases 19-44) ──────────────────────────"
    [[ -f "$s2/summary.txt" ]] && cat "$s2/summary.txt" || echo "(summary not found: $s2)"
} > "$COMBINED"

pass=$(grep -c "^PASS" "$COMBINED" 2>/dev/null || echo 0)
fail=$(grep -c "^FAIL" "$COMBINED" 2>/dev/null || echo 0)
skip=$(grep -c "^SKIP" "$COMBINED" 2>/dev/null || echo 0)

{
    echo
    echo "── TOTAL ────────────────────────────────────────────────────"
    echo "PASS=$pass  FAIL=$fail  SKIP=$skip"
} >> "$COMBINED"

echo
cat "$COMBINED"
echo
echo "combined summary: $COMBINED"
echo "per-case logs:    $s1/  and  $s2/"

[[ $rc1 -eq 0 && $rc2 -eq 0 ]] && exit 0 || exit 1
