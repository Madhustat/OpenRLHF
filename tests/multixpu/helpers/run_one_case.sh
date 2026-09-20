#!/usr/bin/env bash
# Run ONE case of the extended suite.
#
# Must live in its own file: the suite's clear_gpus() runs pkill -f over process
# patterns, and if the invoking shell's command line happens to contain one of those
# patterns it kills the caller. Keeping the invocation in a script means the caller's
# command line is just this path.
#
# usage: run_one_case.sh <case_id>
set -uo pipefail
CASE=${1:?need a case id}
REPO=/home/sdp/madhu/OpenRLHF-fresh
V=/home/sdp/venvs/openrlhf-xccl-auto-detect-213
export LD_LIBRARY_PATH="$V/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export PYTHON=$V/bin/python
export RAY=$V/bin/ray
export PYTHONPATH=$REPO
export OPENRLHF_DS_TORCH_ADAM=1
cd "$REPO"
bash tests/test_e2e_suite_multigpu_extended.sh "$CASE" > "/tmp/case_${CASE}.out" 2>&1
echo "exit=$?"
grep -E '^(PASS|FAIL) ' "/tmp/case_${CASE}.out" | grep -v filtered | head -3
