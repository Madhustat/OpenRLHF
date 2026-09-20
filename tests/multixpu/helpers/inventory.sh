#!/usr/bin/env bash
# Inventory of every OpenRLHF working tree on this box.
set -uo pipefail
for d in OpenRLHF-fresh wt-exp-multi-gloo wt-upstream-merge-trial OpenRLHF-multi OpenRLHF-1xpu-baseline; do
    p=/home/sdp/madhu/$d
    [[ -d $p ]] || continue
    br=$(git -C "$p" branch --show-current 2>/dev/null)
    head=$(git -C "$p" log --oneline -1 2>/dev/null | cut -c1-9)
    dirty=$(git -C "$p" status --porcelain 2>/dev/null | grep -vc '^??' || true)
    # find whichever remote ref tracks upstream OpenRLHF
    behind="?"
    for ref in upstream/main origin/main; do
        if git -C "$p" rev-parse --verify "$ref" >/dev/null 2>&1; then
            behind=$(git -C "$p" rev-list --count "HEAD..$ref" 2>/dev/null)
            behind="$behind (vs $ref)"
            break
        fi
    done
    printf '%-26s %-34s %-10s dirty=%-4s behind=%s\n' "$d" "$br" "$head" "${dirty:-0}" "$behind"
done
