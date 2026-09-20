#!/usr/bin/env python3
"""Run the 2-GPU extended suite, then root-cause every failure automatically.

Sequence:
  1. run tests/test_e2e_suite_multigpu_extended.sh  (55 cases, GPUs cleared per case)
  2. for each FAIL: classify the cause from that case's log against known signatures
  3. retry each failure ONCE in isolation -- this is what separates a transient
     (ordering / leftover memory / vLLM profiling race) from a real defect
  4. write ROOT_CAUSE.md and FINAL_REPORT.txt

Runs detached, so a dropped network connection does not affect it.

Cases the suite documents as expected non-passes are labelled EXPECTED in the
report rather than counted as regressions -- same treatment as the gloo matrix's
"Unsupported" cells.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/sdp/madhu/OpenRLHF-fresh")
SUITE = REPO / "tests/test_e2e_suite_multigpu_extended.sh"
VENV = Path("/home/sdp/venvs/openrlhf-xccl-auto-detect-213")

# Documented in the suite header as expected to fail / be unsupported on XPU.
EXPECTED_FAIL = {
    "mg_ds_autotp2": "DeepSpeed AutoTP on XPU is unproven",
    "mg_ring_attn2": "ring attention needs a flash-attn kernel",
    "mg_liger": "Liger kernels are CUDA-oriented",
    "mg_deepcompile": "DeepCompile unproven on XPU",
    "mg_sync_with_ray": "alternative weight-sync path, never exercised on XPU",
    "mg_flash_attn2": "no XPU flash-attn2 kernel published for torch 2.13",
    "mg_moe_experts_grouped_mm": "needs a grouped-GEMM kernel for this backend",
    "mg_moe_experts_deepgemm": "DeepGEMM is CUDA-only",
    "mg_moe_aux_loss": "upstream bug: .item() on an int (sft_trainer.py:202)",
}

# First match wins. (regex, root cause)
SIGNATURES = [
    (r"'int' object has no attribute 'item'",
     "UPSTREAM BUG: aux_loss is a plain int but .item() is called on it "
     "(sft_trainer.py:202 / rm_trainer.py:188); the guard tests the coefficient, not the type."),
    (r"KeyError: '.*(router|experts)\.",
     "MoE weight-sync name divergence: transformers runtime parameter names do not match "
     "the checkpoint names vLLM's loader expects for this architecture."),
    (r"has no attribute 'offload_states'",
     "DeepSpeed sleep at stage 0: FP16_UnfusedOptimizer has no offload_states(); "
     "architectural, not a defect."),
    (r"Offloading is supported only for DeepSpeed FusedAdam",
     "DeepSpeed stage-3 offload asserts FusedAdam; the FusedAdam-capability fallback "
     "should have prevented this -- verify that patch is present."),
    (r"UR_RESULT_ERROR_DEVICE_LOST",
     "Level-Zero device context lost in a oneCCL collective (P2P path). "
     "Expect only if CCL_*=direct is off."),
    (r"Available KV cache memory: -",
     "vLLM KV cache sized negative: gpu_memory_utilization too low for this topology."),
    (r"Error in memory profiling",
     "vLLM memory-profiling race during colocated init (historically transient)."),
    (r"(OutOfMemoryError|out of memory|XPU out of memory)",
     "Device OOM: this configuration does not fit in 2x16 GB."),
    (r"Segmentation fault",
     "SIGSEGV (historically oneCCL worker thread / Level-Zero)."),
    (r"ModuleNotFoundError|ImportError",
     "Missing dependency for this feature on this platform."),
    (r"chat_template is not set",
     "Model ships no tokenizer.chat_template but --data.apply_chat_template was passed."),
    (r"unrecognized arguments|invalid choice",
     "CLI flag or value not accepted by this version -- the case needs updating."),
    (r"AssertionError",
     "An upstream assertion rejected this configuration (often a structurally invalid combo)."),
    (r"(RuntimeError|ValueError|TypeError|AttributeError)",
     "Runtime error -- see the excerpt; not matched by a known signature."),
]


def env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{VENV}/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    e["PYTHON"] = str(VENV / "bin/python")
    e["RAY"] = str(VENV / "bin/ray")
    e["PYTHONPATH"] = str(REPO)
    e["OPENRLHF_DS_TORCH_ADAM"] = "1"
    return e


def log(msg, fh):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def newest_results():
    dirs = sorted((REPO / "tests/results").glob("multigpu_extended_*"),
                  key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None


def parse_summary(d):
    out = []
    f = d / "summary.txt"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        m = re.match(r"(PASS|FAIL|SKIP)\s+(\S+)\s+—\s*(.*)", line)
        if m:
            out.append(dict(status=m.group(1), case=m.group(2), detail=m.group(3)))
    return out


def classify(case, d):
    logf = d / f"{case}.log"
    if not logf.exists():
        return "No log file produced.", ""
    text = logf.read_text(errors="replace")
    for pat, cause in SIGNATURES:
        m = re.search(pat, text)
        if m:
            excerpt = ""
            for ln in text.splitlines():
                if m.group(0)[:40] in ln:
                    excerpt = ln.strip()[:200]
                    break
            return cause, excerpt
    return "Unclassified -- no known signature matched.", ""


def main():
    rep = REPO / "tests/results"
    rep.mkdir(parents=True, exist_ok=True)
    fh = (rep / f"orchestrator_{time.strftime('%Y%m%d_%H%M%S')}.log").open("w")
    log("=== 2-GPU extended suite + automatic root-cause analysis ===", fh)
    log(f"suite: {SUITE}", fh)

    t0 = time.time()
    p = subprocess.run(["bash", str(SUITE)], cwd=REPO, env=env(),
                       capture_output=True, text=True)
    log(f"suite finished rc={p.returncode} in {(time.time()-t0)/60:.0f} min", fh)

    d = newest_results()
    if not d:
        log("!! no results directory produced", fh)
        return 1
    log(f"results: {d}", fh)

    rows = parse_summary(d)
    fails = [r for r in rows if r["status"] == "FAIL"]
    log(f"pass1: {sum(r['status']=='PASS' for r in rows)} PASS, {len(fails)} FAIL, "
        f"{sum(r['status']=='SKIP' for r in rows)} SKIP of {len(rows)}", fh)

    # ---- root cause + one isolated retry per failure
    findings = []
    for r in fails:
        case = r["case"]
        cause, excerpt = classify(case, d)
        log(f"FAIL {case}: {cause[:70]}", fh)
        log(f"  retrying {case} in isolation...", fh)
        rp = subprocess.run(["bash", str(SUITE), case], cwd=REPO, env=env(),
                            capture_output=True, text=True)
        rd = newest_results()
        retry = "UNKNOWN"
        for rr in parse_summary(rd):
            if rr["case"] == case and rr["status"] in ("PASS", "FAIL"):
                retry = rr["status"]
        log(f"  retry {case}: {retry}", fh)
        findings.append(dict(case=case, detail=r["detail"], cause=cause,
                             excerpt=excerpt, retry=retry,
                             expected=case in EXPECTED_FAIL,
                             expected_why=EXPECTED_FAIL.get(case, "")))

    # ---- reports
    rc = d / "ROOT_CAUSE.md"
    L = ["# Extended 2-GPU suite — failures, root causes, retry outcomes", "",
         f"- Suite: `{SUITE.name}`", f"- Results: `{d.name}`",
         f"- Repo: `{REPO}` (latest upstream + our changes)", ""]
    if not findings:
        L.append("No failures.")
    else:
        real = [f for f in findings if not f["expected"]]
        exp = [f for f in findings if f["expected"]]
        flaky = [f for f in findings if f["retry"] == "PASS"]
        L += [f"**{len(findings)} failures: {len(exp)} expected, {len(real)} unexpected, "
              f"{len(flaky)} passed on retry (transient).**", "",
              "| Case | Expected? | Retry | Root cause |", "|---|---|---|---|"]
        for f in findings:
            L.append(f"| `{f['case']}` | {'yes — ' + f['expected_why'] if f['expected'] else '**NO**'} "
                     f"| **{f['retry']}** | {f['cause']} |")
        L += ["", "## Evidence", ""]
        for f in findings:
            L += [f"### {f['case']}", f"- retry: **{f['retry']}**",
                  f"- cause: {f['cause']}"]
            if f["excerpt"]:
                L += ["```", f["excerpt"], "```"]
            L.append("")
    rc.write_text("\n".join(L) + "\n")
    log(f"root cause report -> {rc}", fh)

    final = d / "FINAL_REPORT.txt"
    with final.open("w") as o:
        o.write("2-GPU EXTENDED SUITE — FINAL REPORT\n")
        o.write(f"generated {time.ctime()}\nresults {d}\n")
        o.write("=" * 78 + "\n\n")
        o.write(f"{'CASE':<34}{'RESULT':<8}DETAIL\n{'-'*34}{'-'*8}{'-'*30}\n")
        for r in rows:
            o.write(f"{r['case']:<34}{r['status']:<8}{r['detail'][:70]}\n")
        o.write(f"\nTOTALS: {sum(r['status']=='PASS' for r in rows)} PASS, "
                f"{len(fails)} FAIL, {sum(r['status']=='SKIP' for r in rows)} SKIP\n")
        if findings:
            o.write(f"\nRetry recovered: {sum(f['retry']=='PASS' for f in findings)}\n")
            o.write(f"Unexpected failures: {sum(not f['expected'] for f in findings)}\n")
        o.write(f"\nRoot causes: {rc}\n")
    log(f"FINAL REPORT -> {final}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
