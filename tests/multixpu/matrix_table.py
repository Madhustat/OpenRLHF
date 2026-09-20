#!/usr/bin/env python3
"""Render the 80-case gloo matrix as the stage-column table, straight from results.jsonl.

Rows are the 20 valid scenarios (X1-X4, X9-X24); the four right-hand columns are the
DeepSpeed stages. Every cell is a short code plus the step count; the code is classified
from the case's own train.log signatures, not hardcoded per case, so cases that have not
run yet simply render as "-" and fill in on the next refresh.

Usage:  python matrix_table.py <run_dir> [-o OUT.md]
Default output is <run_dir>/MATRIX_TABLE.md.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Scenario axes, transcribed from SCENARIOS in run_gloo_matrix.py:210-244. Kept as a literal
# table rather than imported so this renderer never perturbs the harness module.
_SEP = ("XPU 0", "XPU 1", "Actor/vLLM separated")
_A2 = "Actor rank 0 on XPU 0; rank 1 on XPU 1"
_A2C2 = "Actor and Critic ranks span XPU 0 and XPU 1"
_V2E = "One TP1 engine per XPU"
_VTP2 = "TP rank 0 on XPU 0; rank 1 on XPU 1"
_COLO = "Fully colocated"

# id, actor_ws, critic_ws, vllm layout, actor placement, vllm placement, colocation, sleep
SCEN = [
    ("X1", 1, 1, "1 engine x TP1", *_SEP, "Both On"),
    ("X2", 1, 1, "1 engine x TP1", *_SEP, "vLLM On / DS Off"),
    ("X3", 1, 1, "1 engine x TP1", *_SEP, "vLLM Off / DS On"),
    ("X4", 1, 1, "1 engine x TP1", *_SEP, "Both Off"),
    ("X9", 2, 1, "2 engines x TP1", _A2, _V2E, _COLO, "Both On"),
    ("X10", 2, 1, "2 engines x TP1", _A2, _V2E, _COLO, "vLLM On / DS Off"),
    ("X11", 2, 1, "2 engines x TP1", _A2, _V2E, _COLO, "vLLM Off / DS On"),
    ("X12", 2, 1, "2 engines x TP1", _A2, _V2E, _COLO, "Both Off"),
    ("X13", 2, 1, "1 engine x TP2", _A2, _VTP2, _COLO, "Both On"),
    ("X14", 2, 1, "1 engine x TP2", _A2, _VTP2, _COLO, "vLLM On / DS Off"),
    ("X15", 2, 1, "1 engine x TP2", _A2, _VTP2, _COLO, "vLLM Off / DS On"),
    ("X16", 2, 1, "1 engine x TP2", _A2, _VTP2, _COLO, "Both Off"),
    ("X17", 2, 2, "2 engines x TP1", _A2C2, _V2E, _COLO, "Both On"),
    ("X18", 2, 2, "2 engines x TP1", _A2C2, _V2E, _COLO, "vLLM On / DS Off"),
    ("X19", 2, 2, "2 engines x TP1", _A2C2, _V2E, _COLO, "vLLM Off / DS On"),
    ("X20", 2, 2, "2 engines x TP1", _A2C2, _V2E, _COLO, "Both Off"),
    ("X21", 2, 2, "1 engine x TP2", _A2C2, _VTP2, _COLO, "Both On"),
    ("X22", 2, 2, "1 engine x TP2", _A2C2, _VTP2, _COLO, "vLLM On / DS Off"),
    ("X23", 2, 2, "1 engine x TP2", _A2C2, _VTP2, _COLO, "vLLM Off / DS On"),
    ("X24", 2, 2, "1 engine x TP2", _A2C2, _VTP2, _COLO, "Both Off"),
]
STAGES = [0, 1, 2, 3]

LEGEND = [
    ("PASS 5/5", "All 13 pass criteria met, 5/5 steps, gloo weight sync verified fresh on both XPUs"),
    ("BLOCK (ASYNC)", "Rejected before launch: train_ppo_ray.py:686 asserts `not args.vllm.enable_sleep` "
                      "under --train.async_enable"),
    ("FAIL (A0)", "Bug A at stage 0: AttributeError: 'FP16_UnfusedOptimizer' object has no "
                  "attribute 'offload_states'"),
    ("BLOCK (A3)", "Bug A at stage 3: stage3.py:3285 AssertionError: Offloading is supported only for "
                   "DeepSpeed FusedAdam"),
    ("FAIL (HANG)", "Actor finishes the first train epoch, then the driver blocks forever in "
                    "ray::CoreWorker::Get(); killed at the no-progress cutoff. No weight-broadcast "
                    "marker is ever reached"),
    ("FAIL (PROF)", "vLLM EngineCore aborted: `AssertionError: Error in memory profiling. Initial free "
                    "memory X, current free memory Y` -- free memory GREW mid-profile because the "
                    "colocated DeepSpeed engine released its offloaded state"),
    ("FAIL (SEGV)", "Fatal Python error: Segmentation fault in PolicyModelActor, no Python frame"),
    ("FAIL (KV)", "vLLM EngineCore refused to start: negative `Available KV cache memory` at "
                  "gpu_memory_utilization=0.22 -- sizing limit, not a weight-sync defect"),
    ("FAIL (?)", "Failed with a signature not yet classified; see cases/<id>/train.log"),
    ("-", "Not yet run"),
]

# Ordered: first match wins. (needle, code) -- needle searched in error text then train.log tail.
SIGS = [
    ("not args.vllm.enable_sleep", "BLOCK (ASYNC)"),
    ("Offloading is supported only for DeepSpeed FusedAdam", "BLOCK (A3)"),
    ("has no attribute 'offload_states'", "FAIL (A0)"),
    ("Available KV cache memory: -", "FAIL (KV)"),
    ("Error in memory profiling", "FAIL (PROF)"),
    ("Segmentation fault", "FAIL (SEGV)"),
]

# Only reached when no log signature matched.
HANG = "FAIL (HANG)"


def classify(rec: dict, run_dir: Path) -> str:
    status = rec.get("status", "")
    steps = f"{rec.get('steps_completed', 0)}/{rec.get('target_steps', 5)}"
    if status == "PASS":
        return f"**PASS {steps}**"

    haystack = rec.get("error") or ""
    log = run_dir / "cases" / rec["case_id"] / "train.log"
    if log.exists():
        # The signatures all appear in tracebacks near the failure, but stage-3 asserts land
        # early, so scan the whole file rather than a tail window.
        haystack += log.read_text(errors="replace")
    # A dead vLLM EngineCore also trips the harness's no-progress cutoff (the driver is left
    # blocked in ray.get), so the log signatures must be consulted BEFORE trusting
    # failure_phase == "hang / timeout" -- otherwise engine-init failures masquerade as hangs.
    for needle, code in SIGS:
        if needle in haystack:
            return code if code.startswith("BLOCK") else f"{code.split(' (')[0]} {steps} ({code.split('(')[1]}"
    if rec.get("failure_phase") == "hang / timeout":
        return f"FAIL {steps} (HANG)"
    return f"{status} {steps} (?)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    run_dir = a.run_dir
    out = a.out or run_dir / "MATRIX_TABLE.md"

    recs = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    by = {(r["scenario"], r["stage"]): r for r in recs}
    tally = collections.Counter(r["status"] for r in recs)

    L = []
    L.append("# Gloo weight-sync validation matrix -- 20 scenarios x 4 DeepSpeed stages\n")
    L.append(f"- Run id: `{run_dir.name}`")
    L.append(f"- Progress: **{len(recs)}/80** executed")
    L.append(f"- Tally: PASS {tally['PASS']} / FAIL {tally['FAIL']} / BLOCKED {tally['BLOCKED']}")
    L.append("- Backend: gloo only (CPU-staged actor -> vLLM weight broadcast); 2x Intel Arc Pro B70")
    L.append("- X5-X8 declared invalid and not run: critic world size can never exceed actor world "
             "size on a 2-XPU box (see `_X5_8_REASON` in run_gloo_matrix.py)\n")

    hdr = ("| ID | Actor WS | Critic WS | vLLM layout | Actor placement | vLLM placement "
           "| Colocation mode | Sleep | Backend | Cross-XPU transfer "
           "| Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |")
    L.append(hdr)
    L.append("|" + "---|" * 14)
    for sid, aws, cws, layout, aplace, vplace, colo, sleep in SCEN:
        cells = []
        for z in STAGES:
            r = by.get((sid, z))
            cells.append(classify(r, run_dir) if r else "-")
        L.append(f"| {sid} | {aws} | {cws} | {layout} | {aplace} | {vplace} | {colo} | {sleep} "
                 f"| Gloo | Yes | " + " | ".join(cells) + " |")

    L.append("\n## Code legend\n")
    L.append("| Code | Meaning |")
    L.append("|---|---|")
    for code, meaning in LEGEND:
        L.append(f"| {code} | {meaning} |")

    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(recs)}/80)")


if __name__ == "__main__":
    main()
