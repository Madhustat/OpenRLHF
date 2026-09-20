#!/usr/bin/env python3
"""Unattended driver: full 80-case gloo matrix -> table -> root-cause -> retry -> final table.

Runs against GLOO_MATRIX_REPO (default: the latest-upstream tree) with oneCCL
CCL_*=direct LEFT ON, which the 2026-09-19 A/B proved is required at actor_ws=2:
defaults give UR_RESULT_ERROR_DEVICE_LOST at BOTH ZeRO-2 and ZeRO-3, direct passes both.

Why retries are not `--resume`: resume skips every case_id already in results.jsonl,
including failures. So each failure is retried as its own targeted run and the outcome
is merged over the original record for the final table.

Outputs, all under results/<main_run>/:
    MATRIX_TABLE.md              table from the first pass
    ../<main_run>_final/MATRIX_TABLE.md   table after retries  (the one to read)
    ROOT_CAUSE.md                per-failure root cause + retry outcome
    ORCHESTRATOR.log             what this script did
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SUITE = Path(__file__).resolve().parent
RESULTS = SUITE / "results"
PY = Path("/home/sdp/venvs/openrlhf-xccl-auto-detect-213/bin/python")
REPO = os.environ.get("GLOO_MATRIX_REPO", "/home/sdp/madhu/OpenRLHF-fresh")

# Known failure signatures -> short root cause. Ordered: first match wins.
SIGNATURES = [
    ("UR_RESULT_ERROR_DEVICE_LOST", "Level-Zero device context lost in a oneCCL collective "
                                    "(P2P path). Expected only if CCL_*=direct is off."),
    ("has no attribute 'offload_states'", "DeepSpeed sleep needs FusedAdam; torch AdamW in use "
                                          "(stage 0 / unfused optimizer path)."),
    ("Offloading is supported only for DeepSpeed FusedAdam",
     "DeepSpeed stage-3 offload_states asserts FusedAdam; our partial-sleep fallback "
     "should have avoided this -- check the FusedAdam-capability patch is present."),
    ("Available KV cache memory: -", "vLLM KV cache sized negative: gpu_memory_utilization too "
                                     "low for this topology."),
    ("Error in memory profiling", "vLLM memory-profiling race during colocated init."),
    ("Segmentation fault", "SIGSEGV (historically oneCCL worker thread / Level-Zero)."),
    ("out of memory", "Device OOM."),
    ("Ray advertises only", "Ray did not see both XPUs at start."),
]


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def child_env():
    e = dict(os.environ)
    e["GLOO_MATRIX_REPO"] = REPO
    e["OPENRLHF_DS_TORCH_ADAM"] = "1"
    e.pop("GLOO_MATRIX_CCL_DEFAULT", None)   # ensure CCL_*=direct stays ON
    return e


def run_matrix(args, fh):
    cmd = [str(PY), str(SUITE / "run_gloo_matrix.py")] + args
    log(f"$ {' '.join(cmd[1:])}", fh)
    p = subprocess.run(cmd, cwd=SUITE, env=child_env(), capture_output=True, text=True)
    tail = "\n".join(p.stdout.splitlines()[-6:])
    log(f"  exit={p.returncode}\n{tail}", fh)
    return p


def newest_run(tag):
    cands = sorted(RESULTS.glob(f"run_*_{tag}"), key=lambda d: d.stat().st_mtime)
    return cands[-1] if cands else None


def load(run_dir):
    f = run_dir / "results.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def root_cause(rec, run_dir):
    log_f = run_dir / "cases" / rec["case_id"] / "train.log"
    text = log_f.read_text(errors="replace") if log_f.exists() else ""
    for sig, cause in SIGNATURES:
        if sig in text or sig in (rec.get("error") or ""):
            return cause, sig
    if rec.get("failure_phase", "").startswith("STEP 2"):
        return "Configuration invalid for this topology (not launched).", "config"
    if "hang" in (rec.get("failure_phase") or "").lower():
        return "Hang / case timeout with no crash signature.", "hang"
    return "Unclassified -- see train.log.", "?"


def render(run_dir, fh):
    p = subprocess.run([str(PY), str(SUITE / "matrix_table.py"), str(run_dir)],
                       cwd=SUITE, capture_output=True, text=True)
    log(f"  table -> {run_dir/'MATRIX_TABLE.md'} (exit={p.returncode})", fh)
    if p.returncode:
        log(f"  stderr: {p.stderr[-400:]}", fh)


def main():
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    orch_log = RESULTS / f"orchestrator_{stamp}.log"
    fh = orch_log.open("w")
    log(f"repo under test: {REPO}", fh)
    log("oneCCL: CCL_*=direct ON (A/B proved defaults fail at actor_ws=2)", fh)

    # ---- pass 1: the full matrix
    run_matrix(["--stages", "0,1,2,3", "--tag", "latest_full80"], fh)
    main_run = newest_run("latest_full80")
    if not main_run:
        log("!! no run dir produced; aborting", fh)
        return 1
    log(f"main run: {main_run.name}", fh)
    render(main_run, fh)

    recs = load(main_run)
    fails = [r for r in recs if r["status"] == "FAIL"]
    log(f"pass 1: {len(recs)} cases, {sum(r['status']=='PASS' for r in recs)} PASS, "
        f"{len(fails)} FAIL, {sum(r['status']=='BLOCKED' for r in recs)} BLOCKED", fh)

    # ---- root cause + one targeted retry per failure
    report, retried = [], {}
    for r in fails:
        cid = r["case_id"]
        cause, sig = root_cause(r, main_run)
        log(f"FAIL {cid}: {sig} -> retrying once", fh)
        scen, stage = cid.split("-Z")
        tag = f"retry_{cid.replace('-', '')}"
        run_matrix(["--only", scen, "--stages", stage, "--tag", tag], fh)
        rd = newest_run(tag)
        new = load(rd)[0] if rd and load(rd) else None
        outcome = new["status"] if new else "NO RESULT"
        log(f"  retry {cid}: {outcome}", fh)
        report.append(dict(case=cid, first=r["status"], cause=cause, signature=sig,
                           retry=outcome, retry_run=rd.name if rd else None,
                           steps_first=r.get("steps_completed"),
                           steps_retry=new.get("steps_completed") if new else None))
        if new and new["status"] == "PASS":
            retried[cid] = (new, rd)

    # ---- merged view: original records, retry outcomes layered on top
    final = RESULTS / f"{main_run.name}_final"
    if final.exists():
        shutil.rmtree(final)
    (final / "cases").mkdir(parents=True)
    merged = []
    for r in recs:
        cid = r["case_id"]
        src = main_run / "cases" / cid
        if cid in retried:
            new, rd = retried[cid]
            merged.append(new)
            src = rd / "cases" / cid
        else:
            merged.append(r)
        dst = final / "cases" / cid
        if src.exists() and not dst.exists():
            dst.symlink_to(src)          # matrix_table classifies from train.log
    (final / "results.jsonl").write_text("".join(json.dumps(m) + "\n" for m in merged))
    for extra in ("environment",):
        s = main_run / extra
        if s.exists():
            (final / extra).symlink_to(s)
    render(final, fh)

    # ---- root-cause report
    rc = main_run / "ROOT_CAUSE.md"
    L = ["# Failure root causes and retry outcomes", "",
         f"- Main run: `{main_run.name}`", f"- Repo: `{REPO}`",
         "- oneCCL: `CCL_*=direct` ON (defaults fail at actor_ws=2 -- measured 2026-09-19)",
         f"- Final merged table: `{final.name}/MATRIX_TABLE.md`", ""]
    if not report:
        L += ["No failures in pass 1."]
    else:
        L += ["| Case | 1st | steps | Root cause | Retry | steps |",
              "|---|---|---|---|---|---|"]
        for d in report:
            L.append(f"| {d['case']} | {d['first']} | {d['steps_first']}/5 | {d['cause']} "
                     f"| **{d['retry']}** | {d['steps_retry']}/5 |")
        flaky = [d['case'] for d in report if d['retry'] == 'PASS']
        hard = [d['case'] for d in report if d['retry'] != 'PASS']
        L += ["", f"**Passed on retry (order/transient): {len(flaky)}** "
                  f"{', '.join(flaky) if flaky else '-'}",
              "", f"**Still failing: {len(hard)}** {', '.join(hard) if hard else '-'}"]
    rc.write_text("\n".join(L) + "\n")
    log(f"root cause report -> {rc}", fh)

    fin = load(final)
    log(f"FINAL: {sum(r['status']=='PASS' for r in fin)} PASS, "
        f"{sum(r['status']=='FAIL' for r in fin)} FAIL, "
        f"{sum(r['status']=='BLOCKED' for r in fin)} BLOCKED of {len(fin)}", fh)
    log(f"READ THIS: {final/'MATRIX_TABLE.md'}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
