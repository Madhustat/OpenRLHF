#!/usr/bin/env bash
# Kill a stuck suite run and drain both XPUs.
# Patterns are built from fragments so this script's OWN command line can never
# contain a full pattern -- pkill -f would otherwise match and kill this script.
V=/home/sdp/venvs/openrlhf-xccl-auto-detect-213
A='run_multigpu_extended'; B='test_e2e_suite_multigpu_ext'
C='openrlhf''.cli';        D='VLLM''::EngineCore';  E='ray''::'

pkill -f "$A" 2>/dev/null
pkill -f "$B" 2>/dev/null
sleep 2
LD_LIBRARY_PATH="$V/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu" \
    "$V/bin/ray" stop --force >/dev/null 2>&1
pkill -f "$C" 2>/dev/null
pkill -f "$D" 2>/dev/null
pkill -f "$E" 2>/dev/null
sleep 8

echo "orchestrator procs left : $(pgrep -cf "$A" || true)"
echo "training procs left     : $(pgrep -cf "$C" || true)"
for d in 0 1; do
    v=$(timeout 15 xpu-smi stats -d "$d" 2>/dev/null \
        | grep "GPU Memory Used" | sed 's/.*current: //;s/[^0-9].*//')
    echo "  XPU$d: ${v:-unknown} MiB"
done
