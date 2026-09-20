#!/usr/bin/env bash
# Status of the extended-suite run.
#
# Lives in a file because the suite's clear_gpus() pkills on patterns like
# "openrlhf.cli", "ray::" and "VLLM::EngineCore" between every case. Any shell whose
# command line contains one of those gets killed mid-check. A script's command line
# is just its own path, so it is immune.
set -uo pipefail
REPO=/home/sdp/madhu/OpenRLHF-fresh
# assembled so this file's own cmdline never holds the full literal
ORCH='run_multigpu''_extended_with_rca'

P=$(pgrep -f "$ORCH" | head -1 || true)
if [[ -n "$P" ]]; then
    echo "STATUS : RUNNING   pid=$P  parent=$(ps -o ppid= -p "$P" | tr -d ' ')  elapsed=$(ps -o etime= -p "$P" | tr -d ' ')"
else
    echo "STATUS : not running"
fi

D=$(ls -1dt "$REPO"/tests/results/multigpu_extended_* 2>/dev/null | head -1)
[[ -z "$D" ]] && { echo "no results dir yet"; exit 0; }
echo "RESULTS: $D"

done_n=$(grep -cE '^(PASS|FAIL)' "$D/summary.txt" 2>/dev/null || true)
pass_n=$(grep -c '^PASS' "$D/summary.txt" 2>/dev/null || true)
fail_n=$(grep -c '^FAIL' "$D/summary.txt" 2>/dev/null || true)
echo "PROGRESS: ${done_n:-0}/55 done   PASS=${pass_n:-0}  FAIL=${fail_n:-0}"

# stall detection: how long since the newest case log was written
newest=$(ls -t "$D"/*.log 2>/dev/null | head -1)
if [[ -n "$newest" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
    echo "LAST I/O: $((age/60))m ${age}s ago on $(basename "$newest")"
    (( age > 900 )) && echo "  !! WARNING: no output for >15 min — possible stall"
fi

echo "--- recent ---"
tail -4 "$D/run.log" 2>/dev/null || true
if [[ "${fail_n:-0}" -gt 0 ]]; then
    echo "--- failures so far ---"
    grep '^FAIL' "$D/summary.txt" | cut -c1-150
fi
