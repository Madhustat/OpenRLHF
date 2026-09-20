#!/usr/bin/env python3
"""
OpenRLHF Actor -> vLLM weight-synchronization validation suite, GLOO only, two Intel XPUs.
Code under test: the openrlfh_exp_multi branch (gloo weight-sync path), torch 2.13 venv.

Runs the 24-scenario x 4-DeepSpeed-stage matrix sequentially. X5..X8 are declared INVALID
(not executed) because Critic WS 2 cannot be scheduled against an Actor-WS-1 placement
group -- see SCENARIOS[...]["invalid_reason"]. That leaves 20 scenarios x 4 stages = 80
executable cases.

Each case:
  preflight/health-gate -> build+validate config -> launch -> parse evidence -> classify
  -> clean up -> verify cleanup -> append to results.jsonl -> print running tally

Everything is derived from real evidence: the child process log, `xpu-smi` sampling taken
while the case runs, and `sudo dmesg` deltas. Fields the framework does not emit are
reported as NOT_INSTRUMENTED rather than guessed.

Usage
-----
  python run_gloo_matrix.py                       # full suite, 80 cases
  python run_gloo_matrix.py --only X9,X10         # subset of scenarios
  python run_gloo_matrix.py --only X9-X12 --stages 2,3
  python run_gloo_matrix.py --resume RUN_ID       # continue a previous suite
  python run_gloo_matrix.py --dry-run             # print the 80 commands, run nothing
  python run_gloo_matrix.py --list                # print the matrix and validity verdicts

Layout of results:
  results/<RUN_ID>/
    SUITE_SUMMARY.md          human-readable matrix + tallies (rewritten after every case)
    results.jsonl             one JSON object per case (machine-readable, resume source)
    progress.log              append-only running tally
    environment/              versions, checksums, xpu discovery, dmesg baseline
    cases/<CASE_ID>/
      command.txt  config.json  train.log  report.txt
      xpu_samples.jsonl  dmesg_delta.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------------------
# Fixed environment for this box. Verified working: torch 2.13.0+xpu / vLLM 0.27.2.dev0 /
# DeepSpeed 0.19.1 / Ray 2.55.0, 2x Intel Arc Pro B70 (PCI 18:00.0 and 54:00.0).
# --------------------------------------------------------------------------------------
# Code under test: the openrlfh_exp_multi branch, checked out as a DETACHED git worktree so
# the user's own OpenRLHF-multi checkout and its branches are never touched. Recreate with:
#   cd /home/sdp/madhu/OpenRLHF-multi
#   git worktree add --detach /home/sdp/madhu/wt-exp-multi-gloo openrlfh_exp_multi
# The worktree must stay at 0 modified / 0 untracked files -- that is why the prompt dataset
# lives in this suite's data/ dir rather than in the worktree's tests/data/.
# Override with GLOO_MATRIX_REPO to test a different tree (e.g. the latest-upstream
# tree /home/sdp/madhu/OpenRLHF-fresh). Default is unchanged.
# Default to the checkout this script lives in (tests/multixpu/ -> repo root), so a fresh
# clone reproduces against ITSELF. The old absolute default silently tested another tree
# that was 40 commits behind, and still reported PASS.
REPO = Path(os.environ.get("GLOO_MATRIX_REPO",
                           str(Path(__file__).resolve().parents[2])))
SUITE_DIR = Path(__file__).resolve().parent
PYSITE = SUITE_DIR / "pysite"   # holds only sitecustomize.py (probe logger enablement)

# torch-2.13 venv (required: gloo weight-sync still runs under the distributed/Ray stack that
# only this build supports): openrlhf-xccl-auto-detect-213 -- torch 2.13.0+xpu, deepspeed
# 0.19.1, ray 2.55.0, vLLM 0.27.2.dev0+g6e448d0ea.d20260907, transformers 5.7.0.
#
# NOT openrlhf-native-torch213-xpu, even though that is the venv that originally validated
# exp_multi: its vLLM is an *editable* install whose finder maps `vllm` to /tmp/vllm_src,
# which has since been wiped, so `import vllm` raises ModuleNotFoundError there. This venv's
# editable install points at /home/sdp/madhu/vllm-xccl-auto-detect-213, which survives, and
# is the SAME vLLM commit (g6e448d0ea) -- only the build date differs (d20260907 vs d20260903).
# Verified: exp_multi @ 27bbcac4 imports cleanly here and resolve_vllm_sync_backend() returns
# gloo for both None (auto) and an explicit "gloo".
VENV = Path("/home/sdp/venvs/openrlhf-xccl-auto-detect-213")
PYTHON = VENV / "bin" / "python"
SITE_PKGS = VENV / "lib" / "python3.13" / "site-packages"
RAY_BIN = VENV / "bin" / "ray"


def set_venv(path):
    """Repoint every venv-derived path. Called from main() before anything launches."""
    global VENV, PYTHON, SITE_PKGS, RAY_BIN
    VENV = Path(path)
    PYTHON = VENV / "bin" / "python"
    SITE_PKGS = VENV / "lib" / "python3.13" / "site-packages"
    RAY_BIN = VENV / "bin" / "ray"

MODEL = "Qwen/Qwen2.5-0.5B"
# Suite-local, NOT REPO/tests/data -- keeps the worktree at 0 untracked files.
PROMPTS = SUITE_DIR / "data" / "gsm8k_train_prompts.jsonl"
REWARD_FN = REPO / "examples" / "python" / "math_reward_func.py"

N_STEPS = 5                # exactly 5 optimizer steps per case
ROLLOUT_BS = 8             # prompts drawn per iteration
TRAIN_BS = 8               # max_steps = max_samples // train_bs -> 40 // 8 = 5
MAX_SAMPLES = N_STEPS * ROLLOUT_BS     # 40
GPU_MEM_UTIL = 0.22      # uniform across every case so sleep is the only variable

# Per-case exception to that uniformity. Keep this dict as small as possible: every entry
# costs like-for-like comparability with earlier sweeps, which all ran at a flat 0.22.
#
# X12-Z0 is the heaviest cell in the matrix: stage 0 keeps optimizer state UNPARTITIONED on
# every actor rank, both sleeps are Off so nothing can be evicted, and two vLLM engines are
# resident. vLLM's share of 0.22 is then smaller than its own weights + activations, so the
# KV cache is sized negative and EngineCore refuses to start. MEASURED 2026-09-20, X12-Z0
# only, everything else held constant:
#     util 0.22 -> KV -5.40 GiB  FAIL
#     util 0.30 -> KV -2.98 GiB  FAIL
#     util 0.40 -> KV +0.05 GiB  PASS 5/5   <- crosses zero here, 0.05 GiB is a knife edge
#     util 0.50 -> KV +3.08 GiB  PASS 5/5   <- chosen: real headroom, not a coincidence
# This is a capacity limit of the topology, NOT a defect: X12-Z1/Z2/Z3 all pass at 0.22
# because ZeRO partitions the optimizer state away. Recorded in the result record as
# gpu_memory_utilization so the table never implies this cell ran at the default.
CASE_GPU_MEM_UTIL = {"X12-Z0": 0.50}

# --------------------------------------------------------------------------------------
# Memory-lean profile (on by default; disable with --no-memory-lean)
# --------------------------------------------------------------------------------------
# Goal: every one of the 80 cases should fail or pass on its *topology*, never on capacity.
# The box has 2x 32656 MiB and the heaviest cases (X17-X24: actor WS2 + critic WS2 + a vLLM
# engine, all colocated) put four training ranks plus KV cache on two cards. Measured
# evidence this is real: X4-Z2 (the LIGHTEST launchable case: actor WS1 + critic WS1) already
# peaked XPU 0 at 32651 / 32656 MiB, and the archived run
# ppo_gae_colocate/attempt1_2gpu_per_role_OOM_at_step7.log OOMed outright at 2 GPUs per role.
#
# LoRA WAS the obvious lever (only adapters carry gradients/optimizer state, and it applies to
# both actor and critic via args.ds.lora.*), but it is DELIBERATELY OFF -- it makes the suite
# ungradeable. MEASURED, X4-Z2 with --ds.lora.rank 8: 12 of 13 criteria passed but
# 8_param_changed FAILED, and it would fail identically for all 80 cases. Cause, from
# ppo_actor.py:438: the freshness probe tracks parameters whose names end in
# ("input_layernorm.weight", "self_attn.q_proj.weight", "lm_head.weight") and stops after 8.
# Under PEFT, q_proj is renamed to `...q_proj.base_layer.weight` / `...q_proj.lora_A.default.
# weight`, so the endswith() never matches it; the 8 slots fill with embed_tokens[FIRST] plus
# input_layernorm of layers 0-6, none of which are LoRA targets ("all-linear") and all of which
# are FROZEN. Their checksums are therefore identical across every sync generation, so
# "did the actor's weights actually change?" becomes unverifiable. Fixing the probe would mean
# editing the repo under test, which is out of scope. So: rank 0, and the memory budget is met
# with the other levers below (measured X4-Z2 peak: 32651 MiB before -> 27138 MiB lean-with-LoRA).
# --lora-rank N re-enables it for a capacity-only experiment where criterion 8 does not matter.
LORA_RANK = 0              # >0 re-enables LoRA, but see above: criterion 8 then cannot pass

# Backend for DeepSpeed's OWN process group (the gradient all-reduce across actor ranks). This
# is NOT the weight-sync backend: --vllm.sync_backend only governs actor -> vLLM. Set to "gloo"
# because every actor_ws=2 case died in the XCCL gradient reduce with UR_RESULT_ERROR_DEVICE_LOST
# while sitting at ~12 GiB of 32.6 -- full evidence and trade-offs in pysite/sitecustomize.py.
# "" restores stock XCCL.
#
# TESTED 2026-09-08 AND REJECTED -- do not re-try "gloo" here. The patch applies correctly (all
# 5 workers logged the override) but the case then dies EARLIER, in init_model_from_pretrained:
#   RuntimeError: No backend type associated with device type xpu
# torch 2.13 registers gloo for cpu/mps only (default_device_backend_map = {'cpu': 'gloo',
# 'cuda': 'nccl', 'xpu': 'xccl', 'mps': 'gloo'}). Forcing the pair "cpu:gloo,xpu:gloo" is
# accepted by init_process_group() but the first XPU collective raises
#   RuntimeError: ProcessGroupGloo::broadcast: unsupported device type xpu
# i.e. ProcessGroupGloo has no XPU tensor support in this build, and DeepSpeed does not stage
# gradients to CPU first. So XCCL is the ONLY backend that can carry DeepSpeed's gradient
# reduce here, and Bug B (actor_ws=2 -> UR_RESULT_ERROR_DEVICE_LOST) cannot be dodged by
# backend substitution. Kept as a knob only because the sitecustomize hook is already written.
DS_PG_BACKEND = ""
LORA_ALPHA = 16
LEAN_MAX_LEN = 384         # total prompt+response budget (default 2048; matrix used 512)
LEAN_MAX_NEW_TOKENS = 128  # caps generation, so KV cache and sequence length both shrink
LEAN_MICRO_BS = 2          # train micro-batch (was 4): fewer concurrent activations
LEAN_GRAD_CKPT = True      # trade recompute for activation memory
CASE_TIMEOUT = 1500       # seconds of wall clock per case
NO_PROGRESS_TIMEOUT = 300 # seconds with zero new log output -> treat as hung
# Lowered 600 -> 300 partway through the final80 run. A healthy case never goes quiet for
# anything like this long: it emits tqdm progress lines continuously and completes all 5 steps
# in 180-250 s total. X10-Z3 hung during vLLM engine init and burned 790 s before being
# declared hung; with ~7 more stage-3 cases at actor_ws=2 likely to hang the same way, the old
# window cost ~1.5 h of pure waiting. Raise it back if a case is ever misjudged as hung while
# still making progress -- the report records total_s, so that would be visible.
CLEANUP_SETTLE = 25       # seconds to let XPU memory drain after ray stop
IDLE_MEM_MIB = 600        # per-XPU "free" threshold for the health gate

XPU_COUNT = 2

# --------------------------------------------------------------------------------------
# Base scenario matrix
# --------------------------------------------------------------------------------------
# mode:
#   "separated" -> Actor+Critic on XPU0, vLLM on XPU1. On two XPUs this is only reachable
#                  with --train.colocate_all --train.async_enable (the shared placement
#                  group is sized from the actor, so vLLM only lands on the *other* card
#                  when async_enable makes train_ppo_ray.py:80 pass pg=None to vLLM).
#   "colocated" -> --train.colocate_all, synchronous. Requires
#                  vllm.num_engines * tensor_parallel_size == actor world size
#                  (train_ppo_ray.py:61-69).
def _s(idx, actor_ws, critic_ws, engines, tp, mode, vllm_sleep, ds_sleep,
       actor_placement, vllm_placement, coloc_label, valid=True, invalid_reason=None):
    return dict(
        id=f"X{idx}", actor_ws=actor_ws, critic_ws=critic_ws, engines=engines, tp=tp,
        mode=mode, vllm_sleep=vllm_sleep, ds_sleep=ds_sleep,
        actor_placement=actor_placement, vllm_placement=vllm_placement,
        coloc_label=coloc_label, backend="gloo", cross_xpu_expected=False,
        valid=valid, invalid_reason=invalid_reason,
    )


_SEP_ACTOR = "XPU 0"
_SEP_VLLM = "XPU 1"
_SEP_LABEL = "Actor/vLLM separated"
_A2 = "Actor rank 0 on XPU 0; Actor rank 1 on XPU 1"
_A2C2 = "Actor and Critic ranks span XPU 0 and XPU 1"
_V2E = "One TP1 engine per XPU"
_VTP2 = "TP rank 0 on XPU 0; TP rank 1 on XPU 1"
_COLO = "Fully colocated"

_X5_8_REASON = (
    "Critic WS 2 with Actor WS 1 is unschedulable. Under --train.colocate_all the shared "
    "placement group is sized from the ACTOR alone (train_ppo_ray.py:53), so it has exactly "
    "1 bundle; RayActorGroup._initiate_actors places worker rank R at "
    "placement_group_bundle_index=R (launcher.py:281), so critic rank 1 requests bundle "
    "index 1 of a 1-bundle group and Ray raises at actor creation. Without "
    "--train.colocate_all the critic gets num_gpus=1 per rank (launcher.py:255), so "
    "actor(1) + critic(2) + vLLM(1) = 4 exclusive GPUs on a 2-XPU box. Both routes fail: "
    "critic world size can never exceed actor world size here. DECLARED INVALID, NOT RUN."
)

SCENARIOS = [
    _s(1, 1, 1, 1, 1, "separated", True, True, _SEP_ACTOR, _SEP_VLLM, _SEP_LABEL),
    _s(2, 1, 1, 1, 1, "separated", True, False, _SEP_ACTOR, _SEP_VLLM, _SEP_LABEL),
    _s(3, 1, 1, 1, 1, "separated", False, True, _SEP_ACTOR, _SEP_VLLM, _SEP_LABEL),
    _s(4, 1, 1, 1, 1, "separated", False, False, _SEP_ACTOR, _SEP_VLLM, _SEP_LABEL),

    _s(5, 1, 2, 1, 1, "separated", True, True, "Actor XPU 0; Critic spans XPU 0+1",
       _SEP_VLLM, "Actor/vLLM separated; Critic shared", False, _X5_8_REASON),
    _s(6, 1, 2, 1, 1, "separated", True, False, "Actor XPU 0; Critic spans XPU 0+1",
       _SEP_VLLM, "Actor/vLLM separated; Critic shared", False, _X5_8_REASON),
    _s(7, 1, 2, 1, 1, "separated", False, True, "Actor XPU 0; Critic spans XPU 0+1",
       _SEP_VLLM, "Actor/vLLM separated; Critic shared", False, _X5_8_REASON),
    _s(8, 1, 2, 1, 1, "separated", False, False, "Actor XPU 0; Critic spans XPU 0+1",
       _SEP_VLLM, "Actor/vLLM separated; Critic shared", False, _X5_8_REASON),

    _s(9, 2, 1, 2, 1, "colocated", True, True, _A2, _V2E, _COLO),
    _s(10, 2, 1, 2, 1, "colocated", True, False, _A2, _V2E, _COLO),
    _s(11, 2, 1, 2, 1, "colocated", False, True, _A2, _V2E, _COLO),
    _s(12, 2, 1, 2, 1, "colocated", False, False, _A2, _V2E, _COLO),

    _s(13, 2, 1, 1, 2, "colocated", True, True, _A2, _VTP2, _COLO),
    _s(14, 2, 1, 1, 2, "colocated", True, False, _A2, _VTP2, _COLO),
    _s(15, 2, 1, 1, 2, "colocated", False, True, _A2, _VTP2, _COLO),
    _s(16, 2, 1, 1, 2, "colocated", False, False, _A2, _VTP2, _COLO),

    _s(17, 2, 2, 2, 1, "colocated", True, True, _A2C2, _V2E, _COLO),
    _s(18, 2, 2, 2, 1, "colocated", True, False, _A2C2, _V2E, _COLO),
    _s(19, 2, 2, 2, 1, "colocated", False, True, _A2C2, _V2E, _COLO),
    _s(20, 2, 2, 2, 1, "colocated", False, False, _A2C2, _V2E, _COLO),

    _s(21, 2, 2, 1, 2, "colocated", True, True, _A2C2, _VTP2, _COLO),
    _s(22, 2, 2, 1, 2, "colocated", True, False, _A2C2, _VTP2, _COLO),
    _s(23, 2, 2, 1, 2, "colocated", False, True, _A2C2, _VTP2, _COLO),
    _s(24, 2, 2, 1, 2, "colocated", False, False, _A2C2, _VTP2, _COLO),
]
BY_ID = {s["id"]: s for s in SCENARIOS}
STAGES = [0, 1, 2, 3]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sh(cmd, timeout=60, check=False):
    """Run a shell command, return (rc, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        if check and p.returncode != 0:
            raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{out}")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"<timeout after {timeout}s>"


# DIAGNOSTIC OVERRIDES (not part of the matrix) ---------------------------------------
# Populated only by --extra-arg / --set-env / --unset-env, which exist so a single case can
# be A/B'd against a control without editing the scenario table. Empty on a normal run, and
# every override is recorded in the per-case report so a diagnostic run is never mistaken
# for a matrix run.
EXTRA_ARGS: list[str] = []
ENV_OVERRIDES: dict[str, str | None] = {}
MEMORY_LEAN = True   # see the memory-lean profile block above; --no-memory-lean turns it off


def child_env(ray_address: str | None = None) -> dict:
    """Environment for the OpenRLHF child process.

    Two groups matter and both are load-bearing on this box:
      * library path -- the base miniforge3/lib on LD_LIBRARY_PATH ships an older
        libur_loader, which makes torch 2.13's libsycl.so.9 fail with
        "undefined symbol: urDeviceWaitExp". Prepending the venv lib wins.
      * oneCCL knobs for 2x Arc Pro B70 -- without these the XCCL broadcast hangs.
    """
    e = dict(os.environ)
    ld = [
        str(VENV / "lib"),
        str(SITE_PKGS / "vllm_xpu_kernels"),
        e.get("LD_LIBRARY_PATH", ""),
    ]
    e["LD_LIBRARY_PATH"] = ":".join(x for x in ld if x)
    # PYSITE first: it holds only sitecustomize.py, which raises the "weight_freshness"
    # logger to INFO inside every Ray worker so the repo's parameter-checksum probes are
    # actually emitted (see pysite/sitecustomize.py). The repo itself is untouched.
    e["PYTHONPATH"] = f"{PYSITE}{os.pathsep}{REPO}"
    e["PYTHONUNBUFFERED"] = "1"
    # The venv's bin must be on PATH, not just inherited from whatever shell launched the
    # suite: DeepSpeed JIT-builds its XPU ops with torch.utils.cpp_extension, which shells out
    # to `ninja -v`. Without this, a Ray worker gets
    #   subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit
    #   status 127  ->  RuntimeError: Error building extension 'fused_adam'
    # (127 = not found, i.e. a PATH problem, NOT the SYCL-op segfault the repo comment warns
    # about). Only relevant when OPENRLHF_DS_TORCH_ADAM is unset so FusedAdam is actually built.
    e["PATH"] = f"{VENV / 'bin'}{os.pathsep}{e.get('PATH', '')}"

    # Ray must not set ONEAPI_DEVICE_SELECTOR inside an already-running worker: on XPU that
    # is too late for Level-Zero, and both roles would bind logical device 0. With this set,
    # OpenRLHF binds each actor to ray.get_gpu_ids()[0] itself (launcher.py:36).
    e["RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR"] = "1"
    e["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0,1"

    # oneCCL / B70
    e.update(
        CCL_ZE_IPC_EXCHANGE="sockets",
        CCL_ATL_TRANSPORT="ofi",
        CCL_ATL_SHM="1",
        CCL_BUFFER_CACHE="0",
        FI_PROVIDER="shm",
        CCL_ZE_CACHE_OPEN_IPC_HANDLES="0",
        CCL_ZE_CACHE_GET_IPC_HANDLES="0",
        CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK="0",
    )

    # THE FIX FOR BUG B (actor world_size 2). oneCCL's default topology-aware algorithms use
    # Level-Zero device-to-device (P2P/IPC) copies between the two B70s. On this box that path
    # loses the device context: every actor_ws=2 case died in the first micro-batch's gradient
    # reduce with `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)` --
    # X9-Z1/Z2/Z3 and X10-Z0/Z1/Z2, at ~12 GiB of 32.6 GiB and with zero kernel-level CAT/reset
    # events, so neither memory nor a hardware wedge (see the sitecustomize note for the full
    # elimination). `direct` makes oneCCL fall back to its ATL transport (ofi/shm, host-staged)
    # instead. MEASURED: X9-Z2 goes FAIL(0/5 steps) -> PASS(5/5 steps, 256 s) with only this
    # change. Cost: gradient collectives stage through host memory, so they are slower than a
    # working P2P path would be -- irrelevant at 0.5B x 5 steps, and there is no working P2P
    # path here to trade against.
    #
    # NOT YET NARROWED: all six were set at once. Allreduce/reduce-scatter are the ones ZeRO
    # actually leans on, so those are the likely load-bearing entries; the rest are cheap
    # insurance. Narrow only if the host-staged path becomes a bottleneck.
    # GLOO_MATRIX_CCL_DEFAULT=1 leaves oneCCL's algorithm selection at its DEFAULT
    # (topology-aware P2P) instead of forcing the host-staged `direct` path below.
    # The `direct` group was measured necessary on the torch-2.12-era stack; re-measure
    # on torch 2.13 + latest upstream before assuming it is still required.
    if os.environ.get("GLOO_MATRIX_CCL_DEFAULT") != "1":
        e.update(
            CCL_ALLREDUCE="direct",
            CCL_REDUCE="direct",
            CCL_REDUCE_SCATTER="direct",
            CCL_ALLGATHER="direct",
            CCL_ALLGATHERV="direct",
            CCL_BROADCAST="direct",
        )

    # DeepSpeed's fused/CPU Adam SYCL ops JIT-segfault on this torch build; use torch AdamW.
    # COST, measured in run full80: this makes DeepSpeed build torch.optim.AdamW, so
    # offload_deepspeed_states() -> optimizer.offload_states() breaks on the ds_sleep axis at
    # BOTH ends of the stage range -- stage 0 raises
    # `'FP16_UnfusedOptimizer' object has no attribute 'offload_states'` and stage 3 raises
    # `AssertionError: Offloading is supported only for DeepSpeed FusedAdam` (stage3.py:3285).
    # Stages 1/2 are unaffected (X3-Z1, X3-Z2 both PASS 5/5). Use --unset-env
    # OPENRLHF_DS_TORCH_ADAM to test whether FusedAdam still JIT-segfaults on this build.
    e["OPENRLHF_DS_TORCH_ADAM"] = "1"
    # Backend for DEEPSPEED's own PG (gradient all-reduce), distinct from the weight-sync
    # backend. See pysite/sitecustomize.py for the measured justification; empty = stock XCCL.
    if DS_PG_BACKEND:
        e["OPENRLHF_DS_PG_BACKEND"] = DS_PG_BACKEND
    # Actor- and vLLM-side parameter checksum probes (ppo_actor.py:434,
    # vllm_worker_wrap.py:58). Propagated into Ray workers by train_ppo_ray.py:34.
    e["OPENRLHF_WEIGHT_PROBE"] = "1"
    e["TOKENIZERS_PARALLELISM"] = "false"
    e["HF_HUB_OFFLINE"] = "1"      # model is already in the local HF cache
    if ray_address:
        e["RAY_ADDRESS"] = ray_address
    for k, v in ENV_OVERRIDES.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    return e


def ray_start():
    """Start a fresh single-node Ray head with the XPU count declared explicitly.

    Ray cannot discover Intel XPUs on this box: its IntelGPUAcceleratorManager
    (ray/_private/accelerators/intel_gpu.py get_current_node_num_accelerators) imports
    `dpctl` and returns 0 when the import fails. dpctl is deliberately NOT installed here --
    it pulls Intel 2026.x runtime libs that conflict with the 2025.3.2 runtime torch 2.13
    pins. So an in-process ray.init() sees GPU=0 and every placement group hangs with
    "No available node types can fulfill resource request {'GPU': 1.0}".

    Declaring `--num-gpus 2` on an external head is the fix, and it is how the archived
    known-passing runs were driven. Returns (ok, address, output).
    """
    env = child_env()
    cmd = (f"{shlex.quote(str(RAY_BIN))} start --head --num-gpus {XPU_COUNT} "
           f"--disable-usage-stats")
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=240, env=env)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return False, None, "<ray start timed out>"
    if p.returncode != 0:
        return False, None, out

    m = re.search(r"ray start --address='([^']+)'", out)
    address = m.group(1) if m else "auto"

    # Confirm the cluster really advertises the GPUs before anything is launched against it.
    # `ray status` returns "No cluster status ... services to start up" for the first few
    # seconds after `ray start` returns, so poll rather than trusting a single reading.
    status, total = "", 0.0
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            p = subprocess.run(f"{shlex.quote(str(RAY_BIN))} status", shell=True,
                               capture_output=True, text=True, timeout=60, env=env)
            status = (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            status = "<ray status timed out>"
        gm = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s+GPU", status)
        total = float(gm.group(2)) if gm else 0.0
        if total >= XPU_COUNT:
            return True, address, out + "\n" + status
        time.sleep(5)
    return False, address, f"Ray advertises only {total} GPU(s):\n{status}\n{out}"


# --------------------------------------------------------------------------------------
# XPU telemetry
# --------------------------------------------------------------------------------------
def xpu_mem_mib(dev: int):
    rc, out = sh(f"timeout 12 xpu-smi stats -d {dev} -j", timeout=20)
    if rc != 0:
        return None
    try:
        d = json.loads(out)
        tiles = d["memory"]["used_mib"]
        return max(float(t["current"]) for t in tiles.values())
    except Exception:
        return None


def xpu_states():
    """Return {dev: state_string} from xpu-smi discovery."""
    rc, out = sh("timeout 15 xpu-smi discovery", timeout=25)
    states, dev = {}, None
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+Device Name:", line)
        if m:
            dev = int(m.group(1))
        m = re.search(r"Device State:\s*(\S+)", line)
        if m and dev is not None:
            states[dev] = m.group(1)
    return states


def xpu_ps():
    """Return [{pid, device, mem_mib}] from `xpu-smi ps` -- real per-process placement."""
    rc, out = sh("timeout 12 xpu-smi ps", timeout=20)
    rows = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$", line)
        if m:
            rows.append(dict(pid=int(m.group(1)), command=m.group(2),
                             device=int(m.group(3)), shr=m.group(4), mem=m.group(5)))
    return rows


class XpuSampler(threading.Thread):
    """Samples per-device memory and per-process device placement while a case runs.

    This is the independent placement evidence: it does not depend on anything the
    framework chooses to print.
    """

    def __init__(self, path: Path, interval=5.0):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self._stop = threading.Event()
        self.peak = {d: 0.0 for d in range(XPU_COUNT)}
        self.devices_seen = set()
        self.pid_device = {}      # pid -> set(devices)

    def run(self):
        with self.path.open("w") as fh:
            while not self._stop.is_set():
                rec = {"t": time.time(), "mem_mib": {}, "ps": []}
                for d in range(XPU_COUNT):
                    v = xpu_mem_mib(d)
                    if v is not None:
                        rec["mem_mib"][str(d)] = v
                        self.peak[d] = max(self.peak[d], v)
                        if v > IDLE_MEM_MIB:
                            self.devices_seen.add(d)
                for row in xpu_ps():
                    rec["ps"].append(row)
                    self.pid_device.setdefault(row["pid"], set()).add(row["device"])
                    self.devices_seen.add(row["device"])
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=30)


# --------------------------------------------------------------------------------------
# Preflight / cleanup
# --------------------------------------------------------------------------------------
def kill_stragglers():
    patterns = [
        "raylet", "gcs_server", "plasma_store", "ray::", "monitor.py",
        "train_ppo_ray", "log_monitor", "dashboard", "runtime_env_agent",
        "VLLM::", "EngineCore",
    ]
    sh(f"{shlex.quote(str(RAY_BIN))} stop --force", timeout=90)
    for p in patterns:
        sh(f"pkill -9 -f {shlex.quote(p)}", timeout=20)
    sh("rm -rf /tmp/ray/session_* 2>/dev/null", timeout=30)


def cleanup(log=print) -> tuple[bool, str]:
    """Full teardown. Returns (ok, detail)."""
    kill_stragglers()
    deadline = time.time() + CLEANUP_SETTLE
    last = {}
    while time.time() < deadline:
        last = {d: xpu_mem_mib(d) for d in range(XPU_COUNT)}
        if all(v is not None and v <= IDLE_MEM_MIB for v in last.values()):
            return True, f"XPU memory drained to {last}"
        time.sleep(4)
    # one more forced sweep before giving up
    kill_stragglers()
    time.sleep(8)
    last = {d: xpu_mem_mib(d) for d in range(XPU_COUNT)}
    if all(v is not None and v <= IDLE_MEM_MIB for v in last.values()):
        return True, f"XPU memory drained after second sweep to {last}"
    return False, f"XPU memory did NOT drain: {last} (threshold {IDLE_MEM_MIB} MiB)"


# --------------------------------------------------------------------------------------
# Wedged-device recovery
#
# Card 0000:54:00.0 on this box intermittently times out a job. The xe driver's response
# depends on debugfs `wedged_mode`:
#   1 (kernel default) -> "Xe has declared device ... as wedged"; the XPU then vanishes from
#      xpu-smi entirely and EVERY later case dies in engine init. One bad case poisons the run.
#   0                  -> attempt a GT/engine reset instead. Verified 2026-09-07: the 8 MiB
#      all_reduce still errors, but it logs a recoverable "Engine reset" and zero
#      "Kernel-submitted job timed out". That is the behaviour the box had on 09-06, the day
#      ZeRO-2 PPO passed 4 steps, so mode 0 is what a passing configuration expects.
#
# If a card wedges anyway, a PCI remove + rescan re-probes it, reloads GuC and clears the
# wedge in ~20s -- no reboot. Safe on this box specifically because the console is driven by
# card1 (the ast BMC chip, DP-1 connected), not by either Arc card, so re-probing an Arc
# cannot kill the display or an SSH session.
# --------------------------------------------------------------------------------------
RE_WEDGED_DEV = re.compile(r"xe (\d{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d).*?(?:as wedged|device wedged)")


def xe_pci_addrs() -> list[str]:
    """PCI addresses of every DRM card bound to the xe driver (i.e. the Arc cards)."""
    addrs = set()
    for card in Path("/sys/class/drm").glob("card*"):
        drv = card / "device" / "driver"
        try:
            if drv.is_symlink() and drv.resolve().name == "xe":
                addrs.add((card / "device").resolve().name)
        except OSError:
            continue
    return sorted(addrs)


def set_wedged_mode(mode: int = 0, log=print) -> dict:
    """Ask xe to reset engines rather than wedge the device. Best-effort; needs sudo."""
    out = {}
    for a in xe_pci_addrs():
        f = f"/sys/kernel/debug/dri/{a}/wedged_mode"
        sh(f"echo {mode} | sudo -n tee {shlex.quote(f)}", timeout=20)
        rc, cur = sh(f"sudo -n cat {shlex.quote(f)}", timeout=20)
        out[a] = cur.strip() if rc == 0 else "<unreadable>"
    log(f"    wedged_mode: {out}")
    return out


def wedged_pci_addrs() -> list[str]:
    """Cards dmesg says xe declared wedged. Only meaningful once a health check has failed:
    the ring buffer keeps old wedge lines, so this answers "which card" and never "is there
    a problem"."""
    rc, out = sh("sudo -n dmesg | tail -n 4000", timeout=45)
    if rc != 0:
        return []
    xe = set(xe_pci_addrs())
    hits = []
    for line in out.splitlines():
        m = RE_WEDGED_DEV.search(line)
        if m and m.group(1) not in hits and m.group(1) in xe:
            hits.append(m.group(1))
    return hits


def recover_xpus(log=print) -> bool:
    """Re-probe wedged Arc cards over PCI. Returns True if all xe cards came back."""
    before = xe_pci_addrs()
    targets = wedged_pci_addrs() or before
    log(f"    recovery: PCI remove+rescan on {targets} (xe cards: {before})")
    kill_stragglers()
    time.sleep(5)
    for a in targets:
        sh(f"echo 1 | sudo -n tee /sys/bus/pci/devices/{a}/remove", timeout=60)
    time.sleep(5)
    sh("echo 1 | sudo -n tee /sys/bus/pci/rescan", timeout=180)
    time.sleep(20)
    after = xe_pci_addrs()
    set_wedged_mode(0, log=log)
    ok = len(after) >= len(before)
    log(f"    recovery: {'OK' if ok else 'INCOMPLETE'} -- xe cards now {after}")
    return ok


def preflight(log=print) -> tuple[bool, dict]:
    """Health gate that runs before every single case.

    Runs the device checks twice: a wedged card is repairable in ~20s via recover_xpus(), and
    without that retry a single wedge would fail every remaining case in the suite.
    """
    info = {"ts": now_iso()}
    for attempt in (1, 2):
        ok, detail = cleanup()
        info["cleanup"] = detail
        if not ok:
            info["verdict"] = "UNHEALTHY: cleanup failed"
            return False, info

        states = xpu_states()
        info["device_states"] = states
        problem = None
        if len(states) != XPU_COUNT:
            problem = f"expected {XPU_COUNT} XPUs, discovery reported {states}"
        else:
            bad = {d: s for d, s in states.items() if s.lower() != "normal"}
            if bad:
                problem = f"device state not normal: {bad}"

        if problem is None:
            break
        if attempt == 1:
            log(f"    health gate failed ({problem}); attempting XPU recovery")
            info["recovery_attempted"] = problem
            info["recovery_ok"] = recover_xpus(log=log)
            continue
        info["verdict"] = f"UNHEALTHY: {problem}"
        return False, info

    info["free_mem_mib"] = {str(d): xpu_mem_mib(d) for d in range(XPU_COUNT)}
    leftovers = [r for r in xpu_ps()]
    info["unexpected_xpu_processes"] = leftovers
    if leftovers:
        info["verdict"] = f"UNHEALTHY: processes still holding XPUs: {leftovers}"
        return False, info

    info["verdict"] = "HEALTHY"
    return True, info


def dmesg_tail(n=4000) -> str:
    rc, out = sh(f"sudo -n dmesg | tail -n {n}", timeout=45)
    return out if rc == 0 else "<dmesg unavailable>"


def dmesg_since(baseline: str) -> str:
    """Lines present now but not in the baseline snapshot."""
    cur = dmesg_tail()
    base = set(baseline.splitlines())
    return "\n".join(l for l in cur.splitlines() if l not in base)


# --------------------------------------------------------------------------------------
# Configuration builder
# --------------------------------------------------------------------------------------
def build_case(sc: dict, stage: int, gmu: float, steps: int):
    """Return (argv, realization) for one case, or (None, realization) if not launchable.

    realization records exactly how the abstract scenario was mapped onto flags, plus any
    reason the requested configuration cannot be expressed. Nothing is silently altered:
    if the spec cannot be honoured, the case is not launched and the reason is reported.
    """
    r = {
        "mode": sc["mode"],
        "flags_rationale": [],
        "blocking_conflict": None,
        "expected_placement": {
            "actor": sc["actor_ws"], "critic": sc["critic_ws"],
            "engines": sc["engines"], "tp": sc["tp"],
        },
    }
    max_samples = steps * ROLLOUT_BS

    argv = [
        str(PYTHON), "-m", "openrlhf.cli.train_ppo_ray",
        "--actor.num_nodes", "1",
        "--actor.num_gpus_per_node", str(sc["actor_ws"]),
        "--critic.num_nodes", "1",
        "--critic.num_gpus_per_node", str(sc["critic_ws"]),
        "--actor.model_name_or_path", MODEL,
        "--critic.model_name_or_path", MODEL,
        "--vllm.num_engines", str(sc["engines"]),
        "--vllm.tensor_parallel_size", str(sc["tp"]),
        "--vllm.sync_backend", "gloo",
        "--vllm.gpu_memory_utilization", str(gmu),
        "--vllm.enforce_eager",
        "--reward.remote_url", str(REWARD_FN),
        "--data.prompt_dataset", str(PROMPTS),
        "--data.input_key", "prompt",
        "--data.label_key", "label",
        "--data.apply_chat_template",
        "--data.max_len", str(LEAN_MAX_LEN if MEMORY_LEAN else 512),
        "--data.max_samples", str(max_samples),
        "--train.batch_size", str(TRAIN_BS),
        "--train.micro_batch_size", str(LEAN_MICRO_BS if MEMORY_LEAN else 4),
        "--train.max_epochs", "1",
        "--train.num_episodes", "1",
        "--algo.advantage.estimator", "gae",
        "--algo.kl.init_coef", "0",
        "--rollout.n_samples_per_prompt", "1",
        "--rollout.batch_size", str(ROLLOUT_BS),
        "--ds.zero_stage", str(stage),
        "--ds.attn_implementation", "sdpa",
        "--ckpt.save_steps", "-1",
        "--ckpt.output_dir", f"/tmp/gloo_matrix/{sc['id']}-Z{stage}",
        "--eval.steps", "-1",
        "--logger.logging_steps", "1",
    ]

    # ---- memory-lean profile ----
    if MEMORY_LEAN:
        lean = []
        if LORA_RANK > 0:
            lean += ["--ds.lora.rank", str(LORA_RANK), "--ds.lora.alpha", str(LORA_ALPHA)]
        lean += ["--rollout.max_new_tokens", str(LEAN_MAX_NEW_TOKENS)]
        if LEAN_GRAD_CKPT:
            lean += ["--actor.gradient_checkpointing_enable"]
        argv += lean
        r["flags_rationale"].append(
            "MEMORY-LEAN profile active: " + " ".join(lean) +
            f" plus --data.max_len {LEAN_MAX_LEN}, --train.micro_batch_size {LEAN_MICRO_BS}, "
            f"--vllm.gpu_memory_utilization {gmu}. Purpose: make every case fail or pass on its "
            "TOPOLOGY rather than on capacity (X4-Z2, the lightest case, already peaked XPU 0 at "
            "32651/32656 MiB at the original settings). Caveats: LoRA rank "
            f"{LORA_RANK} leaves little gradient/optimizer state to shard, so the ZeRO-stage axis "
            "is weakened, and weight sync now takes the adapter-merge path (ppo_actor.py:526-536) "
            "rather than the full-weight path. Use --no-memory-lean for the original profile."
        )

    # ---- colocation / placement ----
    if sc["mode"] == "colocated":
        argv += ["--train.colocate_all"]
        r["flags_rationale"].append(
            "--train.colocate_all: one shared PACK placement group sized from the actor "
            f"({sc['actor_ws']} bundles); actor/critic/vLLM all request num_gpus=0.2 so they "
            "share physical devices. Satisfies the topology assert at train_ppo_ray.py:61 "
            f"because engines*tp = {sc['engines'] * sc['tp']} == actor world size {sc['actor_ws']}."
        )
    else:
        # Actor+Critic on XPU0, vLLM on XPU1, on a 2-XPU box.
        argv += ["--train.colocate_all", "--train.async_enable"]
        r["flags_rationale"].append(
            "--train.colocate_all --train.async_enable: the ONLY 2-XPU realization of "
            "'Actor/vLLM separated' while the critic still shares the actor's device. "
            "colocate_all packs actor+critic into the 1-bundle PG (XPU 0); async_enable makes "
            "train_ppo_ray.py:80 hand vLLM pg=None so vllm_engine.py builds its own "
            "1-bundle group, which lands on XPU 1. Without colocate_all the critic would "
            "need an exclusive third GPU (launcher.py:255 num_gpus=1). NOTE: this switches "
            "to the asynchronous (off-policy) trainer."
        )
        if sc["vllm_sleep"]:
            r["blocking_conflict"] = (
                "Scenario requires vLLM sleep = On together with Actor/vLLM separation. "
                "train_ppo_ray.py:686 asserts `not args.vllm.enable_sleep` when "
                "--train.async_enable is set, and train_ppo_ray.py:678-680 force "
                "vllm.enable_sleep=False whenever --train.colocate_all is absent. So on two "
                "XPUs, vLLM sleep and Actor/vLLM separation are mutually exclusive: the only "
                "placement that separates them (async) forbids the sleep, and the only mode "
                "that permits the sleep (synchronous colocate_all) puts vLLM on the actor's "
                "own XPU (1 bundle -> both on XPU 0), which is not this scenario and leaves "
                "no cross-XPU path at all. Not launched -- launching it would either crash on "
                "the assert or silently test a different topology."
            )

    # ---- sleep switches (independent, never substituted) ----
    if sc["vllm_sleep"]:
        argv += ["--vllm.enable_sleep"]
    if sc["ds_sleep"]:
        argv += ["--ds.enable_sleep"]
    r["flags_rationale"].append(
        f"vLLM sleep={'ON' if sc['vllm_sleep'] else 'OFF'} "
        f"(--vllm.enable_sleep {'present' if sc['vllm_sleep'] else 'absent'}); "
        f"DeepSpeed sleep={'ON' if sc['ds_sleep'] else 'OFF'} "
        f"(--ds.enable_sleep {'present' if sc['ds_sleep'] else 'absent'}). "
        "--ds.adam_offload is deliberately NOT passed: offload_deepspeed_states() returns "
        "early when adam_offload is set (deepspeed_utils.py), which would make DeepSpeed "
        "sleep a no-op and destroy the ds-sleep axis."
    )

    if EXTRA_ARGS:
        argv += EXTRA_ARGS
        r["flags_rationale"].append(
            "DIAGNOSTIC RUN -- extra args appended by --extra-arg: "
            + " ".join(EXTRA_ARGS)
            + ". This is NOT the matrix configuration."
        )
    if ENV_OVERRIDES:
        r["flags_rationale"].append(
            "DIAGNOSTIC RUN -- env overrides: "
            + ", ".join(f"{k}={'<unset>' if v is None else v}" for k, v in ENV_OVERRIDES.items())
        )
    return argv, r


# --------------------------------------------------------------------------------------
# Log analysis
# --------------------------------------------------------------------------------------
RE_STEP = re.compile(r"Global step (\d+): (\{.*\})")
# Unlike the xpu/xccl-auto-detect branch (where the only progress signal was tqdm), exp_multi
# DOES print "Global step N: {...}" -- verified against the archived gloo PASS run
# ppo_gae_colocate/attempt2_zero_stage2_1gpu_per_role_PASS_10steps.log: 10 step lines, each
# carrying timing/broadcast (~1.12 s, gloo's CPU-staging cost). So on this branch RE_STEP is
# the primary signal and it also supplies criteria 4 and 9 (initial + repeated sync).
# RE_EPISODE stays as the fallback: the outer "Episode" bar advances by ROLLOUT_BS prompts per
# global step, so steps = prompts // ROLLOUT_BS if a build ever stops printing step lines.
RE_EPISODE = re.compile(r"Episode \[\d+/\d+\]:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
RE_BACKEND = re.compile(r"vLLM weight-sync backend:\s*(\S+)")
RE_AUTOSEL = re.compile(r"vLLM weight-sync backend auto-selected:\s*(\S+)")
RE_FALLBACK = re.compile(r"Falling back to gloo")
RE_PG_DEV = re.compile(r"Rank (\d+)\]\s+using GPU (\d+)")
RE_ROLE_PID = re.compile(r"\((PolicyModelActor|CriticModelActor|RolloutRayActor|"
                         r"ReferenceModelActor|PPOTrainer) pid=(\d+)\)")
RE_VLLM_PG = re.compile(r"parallel_state\.py:\d+\] world_size=(\d+) rank=(\d+) local_rank=(\d+)"
                        r".*?backend=(\S+)")
RE_BUNDLE = re.compile(r"creating LLM with bundle_indices=\[([\d,\s]+)\]")
RE_ACTOR_CHK = re.compile(r"ACTOR\s+gen=(\d+)\s+(\S+?)(?:\[FIRST\]|\[LAST\])*\s+chk=(-?[\d.]+)")
RE_VLLM_CHK = re.compile(r"vLLM\s+gen=(\d+)\s+(\S+)\s+chk=(-?[\d.]+)")
RE_SLEEP = re.compile(r"(Sleep mode freed|sleeping tags|wake up tags)")
RE_SLEEP_DOWNGRADE = re.compile(r"Set args\.vllm\.enable_sleep to False")

FATAL_PATTERNS = [
    ("DEVICE_LOST", re.compile(r"UR_RESULT_ERROR_DEVICE_LOST")),
    ("ENGINE_RESET", re.compile(r"Engine reset: engine_class")),
    ("OOM", re.compile(r"(out of memory|OutOfMemoryError|XPU out of memory|"
                       r"UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)", re.I)),
    ("ASSERTION", re.compile(r"^\s*AssertionError", re.M)),
    ("RAY_UNSCHEDULABLE", re.compile(r"(cannot be scheduled right now|"
                                     r"Invalid placement group bundle index|"
                                     r"bundle index \d+ is invalid)", re.I)),
    ("NOT_IMPLEMENTED", re.compile(r"^\s*NotImplementedError", re.M)),
    ("TIMEOUT_COLLECTIVE", re.compile(r"(Watchdog caught collective operation timeout|"
                                      r"NCCL|XCCL).*timeout", re.I)),
]

MODEL_READY = re.compile(r"(Qwen2ForCausalLM|Actor\(|CriticModel\(|"
                         r"\(PolicyModelActor pid=\d+\).*?DeepSpeed)", re.S)


def analyze(log_text: str, sc: dict, stage: int, sampler: XpuSampler | None):
    a: dict = {}

    # --- steps ---
    steps = []
    for m in RE_STEP.finditer(log_text):
        n = int(m.group(1))
        raw = m.group(2)
        d = {}
        # The exponent sign must be inside the alternation: a char class of [\d.eE+] (as the
        # XCCL suite used) truncates "4.5e-05" to "4.5e", which then fails float() and is
        # silently swallowed by the except below -- actor_lr/critic_lr vanished that way.
        for km in re.finditer(r"'([^']+)':\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)", raw):
            try:
                d[km.group(1)] = float(km.group(2))
            except ValueError:
                pass
        steps.append((n, d))
    a["steps_completed"] = max((n for n, _ in steps), default=0)
    a["step_metrics"] = {n: d for n, d in steps}

    # Fallback when the build emits no "Global step" lines (this one does not): derive the count
    # from the Episode bar's prompt counter. Kept as a fallback rather than a replacement so a
    # build that does print explicit step lines is still preferred -- those carry the metrics.
    ep = [(int(m.group(1)), int(m.group(2))) for m in RE_EPISODE.finditer(log_text)]
    if ep:
        a["episode_prompts_seen"] = max(n for n, _ in ep)
        a["episode_prompts_total"] = max(t for _, t in ep)
        if not steps:
            a["steps_completed"] = a["episode_prompts_seen"] // ROLLOUT_BS
            a["steps_source"] = "episode_bar"
    if steps:
        a["steps_source"] = "global_step_lines"
    bcasts = [d.get("timing/broadcast") for _, d in steps if d.get("timing/broadcast") is not None]
    a["broadcast_durations_s"] = bcasts
    a["first_sync_s"] = bcasts[0] if bcasts else None
    a["second_sync_s"] = bcasts[1] if len(bcasts) > 1 else None
    a["losses"] = {
        n: {k: d.get(k) for k in ("policy_loss", "critic_loss", "reward", "actor_grad_norm")}
        for n, d in steps
    }

    # --- effective weight-sync backend ---
    eff = RE_BACKEND.findall(log_text)
    auto = RE_AUTOSEL.findall(log_text)
    a["effective_backend"] = sorted(set(eff)) or sorted(set(auto)) or []
    a["gloo_fallback"] = bool(RE_FALLBACK.search(log_text))
    a["backend_is_xccl"] = any("xccl" in b for b in a["effective_backend"]) and not a["gloo_fallback"]
    # This suite requests gloo EXPLICITLY, so "Falling back to gloo" is not expected and not a
    # failure either way -- gloo is the intended backend. What must be true is that the effective
    # backend really is gloo and not silently something else.
    a["backend_is_gloo"] = (
        any("gloo" in b for b in a["effective_backend"]) or a["gloo_fallback"]
    ) and not a["backend_is_xccl"]

    # --- XCCL sync group: rank -> physical GPU ---
    rank_gpu = {}
    for m in RE_PG_DEV.finditer(log_text):
        rank_gpu[int(m.group(1))] = int(m.group(2))
    a["xccl_rank_to_gpu"] = rank_gpu
    a["xccl_group_world_size"] = (max(rank_gpu) + 1) if rank_gpu else None
    a["xccl_participating_ranks"] = sorted(rank_gpu)
    a["xccl_distinct_devices"] = sorted(set(rank_gpu.values()))

    # --- roles / pids ---
    roles = {}
    for m in RE_ROLE_PID.finditer(log_text):
        roles.setdefault(m.group(1), set()).add(int(m.group(2)))
    a["role_pids"] = {k: sorted(v) for k, v in roles.items()}

    # --- vLLM internal parallel state ---
    a["vllm_internal_pg"] = [
        dict(world_size=int(m.group(1)), rank=int(m.group(2)),
             local_rank=int(m.group(3)), backend=m.group(4))
        for m in RE_VLLM_PG.finditer(log_text)
    ]
    a["vllm_bundle_indices"] = [
        [int(x) for x in m.group(1).replace(" ", "").split(",") if x != ""]
        for m in RE_BUNDLE.finditer(log_text)
    ]

    # --- checksums: actor before/after, vLLM after receive ---
    actor_chk: dict = {}
    for m in RE_ACTOR_CHK.finditer(log_text):
        actor_chk.setdefault(int(m.group(1)), {})[m.group(2)] = float(m.group(3))
    vllm_chk: dict = {}
    for i, m in enumerate(RE_VLLM_CHK.finditer(log_text)):
        vllm_chk.setdefault(m.group(2), []).append(float(m.group(3)))
    a["actor_checksums_by_gen"] = actor_chk
    a["vllm_checksums_by_param"] = vllm_chk

    gens = sorted(actor_chk)
    a["probe_generations"] = gens
    # The probe is env-gated (OPENRLHF_WEIGHT_PROBE=1). Distinguish "probe silent" from
    # "probe ran and showed nothing" -- the two mean completely different things.
    a["weight_probe_active"] = bool(actor_chk) or bool(vllm_chk)

    # A parameter changed if any tracked checksum differs between two broadcast generations.
    changed = []
    if len(gens) >= 2:
        g0, g1 = actor_chk[gens[0]], actor_chk[gens[-1]]
        for k in set(g0) & set(g1):
            if abs(g0[k] - g1[k]) > 1e-6:
                changed.append(k)
    a["actor_params_changed"] = sorted(changed)

    # vLLM receiver freshness. Names travel verbatim from the actor into
    # engine.update_weight (ppo_actor.py _broadcast_param), so they match exactly; fall back
    # to a suffix match for safety. For TP2 the vLLM tensor is a shard of the actor tensor,
    # so exact equality is not applicable -- record the deltas and the reason instead.
    eq, mism, na = [], [], []
    tp_shards = sc["tp"] > 1

    def _actor_val(table, name):
        if name in table:
            return table[name]
        for k, v in table.items():
            if k.endswith(name) or name.endswith(k):
                return v
        return None

    if gens:
        latest = actor_chk[gens[-1]]
        for name, vals in vllm_chk.items():
            ref = _actor_val(latest, name)
            if ref is None:
                na.append({"param": name, "vllm_sums": vals,
                           "reason": "no actor-side checksum tracked for this parameter"})
                continue
            hit = any(abs(v - ref) <= max(1e-3, abs(ref) * 1e-4) for v in vals)
            if hit:
                eq.append(name)
            elif tp_shards:
                na.append({"param": name, "actor_sum": ref, "vllm_sums": vals,
                           "reason": "TP2 shard: vLLM holds a partition of the actor tensor, "
                                     "exact equality N/A",
                           "max_abs_diff": min(abs(v - ref) for v in vals)})
            else:
                mism.append({"param": name, "actor_sum": ref, "vllm_sums": vals,
                             "max_abs_diff": min(abs(v - ref) for v in vals)})
    a["param_equality_matched"] = sorted(eq)
    a["param_equality_mismatched"] = mism
    a["param_equality_not_applicable"] = na

    # --- sleep evidence ---
    a["sleep_markers"] = len(RE_SLEEP.findall(log_text))
    a["vllm_sleep_downgraded"] = bool(RE_SLEEP_DOWNGRADE.search(log_text))

    # --- fatal signatures ---
    fatal = []
    for label, rx in FATAL_PATTERNS:
        m = rx.search(log_text)
        if m:
            line = next((l for l in log_text.splitlines() if m.group(0)[:40] in l), m.group(0))
            fatal.append({"kind": label, "first_line": line.strip()[:400]})
    a["fatal_signatures"] = fatal
    a["traceback_present"] = "Traceback (most recent call last)" in log_text
    a["model_initialized"] = bool(MODEL_READY.search(log_text))

    # --- placement, from xpu-smi sampling (independent of framework output) ---
    if sampler is not None:
        a["peak_xpu_mem_mib"] = {str(d): round(v, 1) for d, v in sampler.peak.items()}
        a["devices_with_activity"] = sorted(sampler.devices_seen)
        a["pid_to_device"] = {str(p): sorted(v) for p, v in sampler.pid_device.items()}
    else:
        a["peak_xpu_mem_mib"] = {}
        a["devices_with_activity"] = []
        a["pid_to_device"] = {}
    return a


def classify(sc, stage, a, rc, timed_out, hung, realization, steps_target):
    """Apply the pass criteria. Returns (status, failure_phase, error_summary, criteria)."""
    if realization.get("blocking_conflict"):
        return ("BLOCKED", "STEP 2: configuration validation",
                realization["blocking_conflict"], {})

    c: dict = {}
    # Placement evidence for gloo MUST come from the xpu-smi sampler, not from RE_PG_DEV.
    # RE_PG_DEV matches ProcessGroupXCCL's "[Rank N] using GPU M", which under gloo weight-sync
    # is emitted by DEEPSPEED's own internal PG, not by the weight-sync group. At actor_ws=1 that
    # yields {0} alone, which shadowed the sampler's correct [0, 1] and failed X4-Z2 even though
    # the run trained 5/5 steps with vLLM demonstrably resident on XPU 1 (10.7 GiB).
    # The sampler is independent of anything the framework prints; prefer it and keep the XCCL
    # rank map as reported-only metadata.
    devs = set(a["devices_with_activity"]) or set(a["xccl_distinct_devices"])

    probe = a["weight_probe_active"]
    c["1_roles_scheduled"] = bool(a["model_initialized"])
    c["2_placement_matches"] = None
    # gloo's weight-sync group is a CPU process group, so it never logs ProcessGroupXCCL's
    # "[Rank N] using GPU M" line -- requiring xccl_group_world_size here (as the XCCL suite
    # does) would fail every case. The gloo equivalent is simply: the effective backend is gloo.
    c["3_sync_pg_init"] = a["backend_is_gloo"]
    c["4_initial_sync"] = bool(a["broadcast_durations_s"]) or bool(a["actor_checksums_by_gen"])
    # Criteria 5, 8 and 10 depend on the checksum probe. If the probe never emitted, they are
    # UNVERIFIABLE (None), not satisfied -- reported as such rather than assumed either way.
    c["5_initial_param_equality"] = (
        (bool(a["param_equality_matched"]) and not a["param_equality_mismatched"]) if probe else None)
    c["6_rollout"] = a["steps_completed"] >= 1
    c["7_optimizer_step"] = a["steps_completed"] >= 1
    c["8_param_changed"] = bool(a["actor_params_changed"]) if probe else None
    c["9_updated_sync"] = len(a["broadcast_durations_s"]) >= 2
    c["10_receivers_updated"] = (
        (bool(a["param_equality_matched"]) and not a["param_equality_mismatched"]) if probe else None)
    c["11_sleep_wake"] = None
    c["12_no_crash"] = (rc == 0) and not a["fatal_signatures"] and not timed_out and not hung
    c["13_cleanup"] = None
    c["steps_target_met"] = a["steps_completed"] >= steps_target

    # placement check
    if sc["mode"] == "colocated":
        want = {0, 1}
        c["2_placement_matches"] = devs == want
        if sc["tp"] > 1 and a["vllm_bundle_indices"]:
            c["2_placement_matches"] = c["2_placement_matches"] and all(
                len(set(b)) == sc["tp"] for b in a["vllm_bundle_indices"])
    else:
        c["2_placement_matches"] = devs == {0, 1}

    # sleep check
    if sc["vllm_sleep"]:
        c["11_sleep_wake"] = a["sleep_markers"] > 0 and not a["vllm_sleep_downgraded"]
    else:
        c["11_sleep_wake"] = not a["vllm_sleep_downgraded"]

    # ---- status ----
    if any(f["kind"] == "RAY_UNSCHEDULABLE" for f in a["fatal_signatures"]):
        return ("UNSCHEDULABLE", "STEP 3: Ray placement",
                a["fatal_signatures"][0]["first_line"], c)

    if (timed_out or hung) and not a["model_initialized"]:
        return ("UNSCHEDULABLE", "STEP 3: Ray placement",
                "No role ever initialized before the timeout: the placement group was never "
                "satisfied (classic 2-XPU over-subscription signature).", c)

    for kind, phase in (("DEVICE_LOST", "STEP 5/6: training or post-update sync"),
                        ("ENGINE_RESET", "STEP 5/6: training or post-update sync")):
        hit = next((f for f in a["fatal_signatures"] if f["kind"] == kind), None)
        if hit:
            return ("FAIL", phase, hit["first_line"], c)

    oom = next((f for f in a["fatal_signatures"] if f["kind"] == "OOM"), None)
    if oom:
        return ("FAIL", "STEP 3/5: memory allocation", oom["first_line"], c)

    for kind in ("ASSERTION", "NOT_IMPLEMENTED"):
        hit = next((f for f in a["fatal_signatures"] if f["kind"] == kind), None)
        if hit:
            return ("BLOCKED", "STEP 2: configuration validation", hit["first_line"], c)

    if timed_out or hung:
        return ("FAIL", "hang / timeout",
                f"No completion within the case budget; steps reached "
                f"{a['steps_completed']}/{steps_target}." +
                (" No new log output before the no-progress cutoff." if hung else ""), c)

    if a["backend_is_xccl"]:
        return ("BLOCKED", "STEP 2: backend selection",
                "Requested gloo but the effective weight-sync backend was xccl -- the case does "
                "not test gloo.", c)

    hard = ["1_roles_scheduled", "2_placement_matches", "3_sync_pg_init", "4_initial_sync",
            "6_rollout", "7_optimizer_step", "8_param_changed", "9_updated_sync",
            "11_sleep_wake", "12_no_crash", "steps_target_met"]
    failed = [k for k in hard if c.get(k) is False]
    if a["param_equality_mismatched"]:
        failed.append("5/10_param_equality")
    if failed:
        return ("FAIL", f"pass criteria unmet: {', '.join(failed)}",
                f"Criteria failed: {failed}. rc={rc}, steps={a['steps_completed']}/{steps_target}.", c)

    if not a["weight_probe_active"]:
        return ("FAIL", "STEP 4/6: weight-probe verification",
                "Every functional criterion passed and training completed "
                f"{a['steps_completed']}/{steps_target} steps, but the checksum probe emitted "
                "no records, so pass criteria 5 (initial parameter equality), 8 (actor "
                "parameter changed) and 10 (receivers hold updated values) could not be "
                "verified. Not reported as PASS: the weight-sync correctness claim is exactly "
                "what this suite exists to prove.", c)

    return ("PASS", "-", "-", c)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def write_report(path: Path, sc, stage, cid, argv, realization, a, status, phase, err,
                 crit, timings, preflight_info, cleanup_detail, dmesg_delta, log_path):
    NI = "NOT_INSTRUMENTED (framework emits no such record)"

    def yn(v):
        return {True: "YES", False: "NO", None: "N/A"}.get(v, str(v))

    vlayout = f"{sc['engines']} engine(s) x TP{sc['tp']}"
    devs = a.get("xccl_distinct_devices") or a.get("devices_with_activity") or []
    tp_ok = (yn(all(len(set(b)) == sc["tp"] for b in a["vllm_bundle_indices"]))
             if sc["tp"] > 1 and a["vllm_bundle_indices"] else ("N/A (TP1)" if sc["tp"] == 1 else NI))
    eng_ok = ("YES" if sc["engines"] == 2 and len(a["vllm_internal_pg"]) >= 2
              else ("N/A (single engine)" if sc["engines"] == 1 else "NO"))

    lines = [
        f"Test Case ID:                    {cid}",
        f"Base Scenario:                   {sc['id']}",
        f"DeepSpeed Stage:                 {'Stage 0' if stage == 0 else f'ZeRO-{stage}'}",
        f"Actor World Size:                {sc['actor_ws']}",
        f"Critic World Size:               {sc['critic_ws']}",
        f"vLLM Layout:                     {vlayout}",
        f"Actor Physical Placement:        requested={sc['actor_placement']} | "
        f"observed XCCL rank->GPU={a['xccl_rank_to_gpu'] or NI}",
        f"vLLM Physical Placement:         requested={sc['vllm_placement']} | "
        f"observed bundle_indices={a['vllm_bundle_indices'] or 'N/A (TP1 path logs none)'}",
        f"Colocation Mode:                 {sc['coloc_label']}",
        f"vLLM Sleep:                      {'On' if sc['vllm_sleep'] else 'Off'}"
        + ("  [WARNING: framework downgraded it to False]" if a["vllm_sleep_downgraded"] else ""),
        f"DeepSpeed Sleep:                 {'On' if sc['ds_sleep'] else 'Off'}",
        f"Weight-Sync Backend:             requested=gloo | effective={a['effective_backend'] or NI}"
        + ("  [FELL BACK TO GLOO]" if a["gloo_fallback"] else ""),
        f"Cross-XPU Transfer Verified:     {yn(len(devs) >= 2)}  (devices carrying the sync group: {devs})",
        "Effective Command:",
        "    " + " \\\n        ".join(shlex.quote(x) for x in argv) if argv else "    <not launched>",
        f"Ray Placement Result:            {'satisfied' if a['model_initialized'] else 'NOT satisfied'}",
        f"XCCL Group World Size:           {a['xccl_group_world_size'] if a['xccl_group_world_size'] is not None else NI}",
        f"Participating Ranks:             {a['xccl_participating_ranks'] or NI}",
        f"Initial Synchronization Result:  {'OK' if crit.get('4_initial_sync') else 'NOT OBSERVED'}",
        f"Initial Parameter Equality:      matched={a['param_equality_matched'] or []} | "
        f"mismatched={a['param_equality_mismatched'] or []} | n/a={a['param_equality_not_applicable'] or []}",
        f"Rollout Result:                  {'OK' if crit.get('6_rollout') else 'NOT OBSERVED'}",
        f"Critic Result:                   critic_loss per step="
        f"{ {k: v.get('critic_loss') for k, v in (a['losses'] or {}).items()} or NI}",
        f"Actor Optimizer-Step Result:     {a['steps_completed']}/{N_STEPS} global steps completed",
        f"Actor Parameter Changed:         {yn(bool(a['actor_params_changed']))} "
        f"{a['actor_params_changed'] or ''}",
        f"Post-Update Synchronization:     {'OK' if crit.get('9_updated_sync') else 'NOT OBSERVED'} "
        f"(broadcasts observed: {len(a['broadcast_durations_s'])})",
        f"Post-Update Parameter Equality:  {'matched' if crit.get('10_receivers_updated') else 'NOT CONFIRMED'}",
        f"Both TP1 Engines Verified:       {eng_ok}",
        f"Both TP2 Ranks Verified:         {tp_ok}",
        f"Sleep Result:                    {yn(crit.get('11_sleep_wake'))} "
        f"(vLLM sleep/wake markers seen: {a['sleep_markers']})",
        f"Wake Result:                     {'OK -- training continued past the first sleep cycle' if a['steps_completed'] >= 2 else 'NOT CONFIRMED'}",
        f"XPU 0 Peak Memory:               {a['peak_xpu_mem_mib'].get('0', NI)} MiB",
        f"XPU 1 Peak Memory:               {a['peak_xpu_mem_mib'].get('1', NI)} MiB",
        f"First Sync Duration:             {a['first_sync_s']} s",
        f"Second Sync Duration:            {a['second_sync_s']} s",
        f"Total Runtime:                   {timings.get('total_s')} s "
        f"(preflight {timings.get('preflight_s')} s, cleanup {timings.get('cleanup_s')} s)",
        f"Final Status:                    {status}",
        f"Failure Phase:                   {phase}",
        f"Error Summary:                   {err}",
        f"Log Location:                    {log_path}",
        "Notes:",
    ]
    for note in realization.get("flags_rationale", []):
        lines.append(f"    - {note}")
    lines.append(f"    - Weight checksum probe: "
                 f"{'ACTIVE' if a.get('weight_probe_active') else 'SILENT (criteria 5/8/10 unverifiable)'}"
                 f" | actor generations={a['probe_generations']}"
                 f" | vLLM params seen={sorted(a['vllm_checksums_by_param'])}")
    lines.append(f"    - Preflight verdict: {preflight_info.get('verdict')}")
    lines.append(f"    - Cleanup after case: {cleanup_detail}")
    lines.append(f"    - Broadcast durations (s): {a['broadcast_durations_s']}")
    lines.append(f"    - Role PIDs: {a['role_pids'] or NI}")
    lines.append(f"    - PID -> physical XPU (xpu-smi ps): {a['pid_to_device'] or NI}")
    lines.append(f"    - vLLM internal parallel state: {a['vllm_internal_pg'] or NI}")
    if a["fatal_signatures"]:
        lines.append(f"    - Fatal signatures: {a['fatal_signatures']}")
    if dmesg_delta.strip():
        kern = [l for l in dmesg_delta.splitlines()
                if re.search(r"xe 0000:|Engine reset|GPU HANG|timed out", l)]
        lines.append(f"    - Kernel (dmesg) lines during this case: {kern or 'none relevant'}")
    lines.append(f"    - Pass-criteria detail: {json.dumps(crit, indent=6)}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Case runner
# --------------------------------------------------------------------------------------
def run_case(sc, stage, run_dir: Path, gmu: float, steps: int, case_timeout: int):
    cid = f"{sc['id']}-Z{stage}"
    cdir = run_dir / "cases" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    log_path = cdir / "train.log"
    t_all = time.time()

    # ---- STEP 1: preflight ----
    t0 = time.time()
    healthy, pinfo = preflight()
    preflight_s = round(time.time() - t0, 1)
    (cdir / "preflight.json").write_text(json.dumps(pinfo, indent=2, default=str))
    if not healthy:
        return dict(case_id=cid, scenario=sc["id"], stage=stage, status="BLOCKED",
                    failure_phase="STEP 1: preflight", error=pinfo["verdict"],
                    suite_abort=True, ts=now_iso())

    # ---- STEP 2: build + validate ----
    argv, realization = build_case(sc, stage, gmu, steps)
    (cdir / "config.json").write_text(json.dumps(
        {"scenario": sc, "stage": stage, "realization": realization,
         "gpu_memory_utilization": gmu, "target_steps": steps}, indent=2))
    (cdir / "command.txt").write_text(
        (" \\\n    ".join(shlex.quote(x) for x in argv) + "\n") if argv else "<not launched>\n")

    if realization.get("blocking_conflict"):
        a = analyze("", sc, stage, None)
        status, phase, err, crit = classify(sc, stage, a, None, False, False, realization, steps)
        write_report(cdir / "report.txt", sc, stage, cid, argv, realization, a, status, phase,
                     err, crit, dict(total_s=round(time.time() - t_all, 1),
                                     preflight_s=preflight_s, cleanup_s=0),
                     pinfo, "not needed (never launched)", "", log_path)
        return dict(case_id=cid, scenario=sc["id"], stage=stage, status=status,
                    failure_phase=phase, error=err, steps_completed=0, launched=False,
                    total_s=round(time.time() - t_all, 1), ts=now_iso())

    # ---- STEP 3: fresh Ray cluster with the XPUs declared ----
    ray_ok, ray_addr, ray_out = ray_start()
    (cdir / "ray_start.txt").write_text(f"ok={ray_ok} address={ray_addr}\n\n{ray_out}")
    if not ray_ok:
        cleanup()
        return dict(case_id=cid, scenario=sc["id"], stage=stage, status="BLOCKED",
                    failure_phase="STEP 3: Ray cluster bring-up",
                    error=f"Could not start a Ray head advertising {XPU_COUNT} GPUs: "
                          f"{ray_out[-600:]}",
                    steps_completed=0, launched=False, suite_abort=True,
                    total_s=round(time.time() - t_all, 1), ts=now_iso())

    # ---- STEP 3-7: launch ----
    dmesg_base = dmesg_tail()
    sampler = XpuSampler(cdir / "xpu_samples.jsonl")
    sampler.start()

    print(f"      launching: stage={stage} actor_ws={sc['actor_ws']} critic_ws={sc['critic_ws']} "
          f"engines={sc['engines']}xTP{sc['tp']} vllm_sleep={sc['vllm_sleep']} ds_sleep={sc['ds_sleep']}")

    timed_out = hung = False
    rc = None
    with log_path.open("w") as lf:
        lf.write(f"# {cid}  {now_iso()}\n# " + " ".join(shlex.quote(x) for x in argv) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(argv, cwd=str(REPO), env=child_env(ray_addr),
                                stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        deadline = time.time() + case_timeout
        last_size, last_growth = 0, time.time()
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            size = log_path.stat().st_size
            if size != last_size:
                last_size, last_growth = size, time.time()
            if time.time() - last_growth > NO_PROGRESS_TIMEOUT:
                hung = True
                break
            if time.time() > deadline:
                timed_out = True
                break
            time.sleep(5)

        if rc is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            rc = proc.poll()

    sampler.stop()
    dmesg_delta = dmesg_since(dmesg_base)
    (cdir / "dmesg_delta.txt").write_text(dmesg_delta)

    log_text = log_path.read_text(errors="replace")
    a = analyze(log_text, sc, stage, sampler)

    # ---- STEP 8: collect + clean up ----
    t0 = time.time()
    cleanup_ok, cleanup_detail = cleanup()
    cleanup_s = round(time.time() - t0, 1)

    status, phase, err, crit = classify(sc, stage, a, rc, timed_out, hung, realization, steps)
    crit["13_cleanup"] = cleanup_ok
    if status == "PASS" and not cleanup_ok:
        status, phase = "FAIL", "STEP 8: cleanup"
        err = cleanup_detail

    timings = dict(total_s=round(time.time() - t_all, 1),
                   preflight_s=preflight_s, cleanup_s=cleanup_s)
    write_report(cdir / "report.txt", sc, stage, cid, argv, realization, a, status, phase, err,
                 crit, timings, pinfo, cleanup_detail, dmesg_delta, log_path)

    rec = dict(case_id=cid, scenario=sc["id"], stage=stage, status=status,
               failure_phase=phase, error=err, rc=rc, timed_out=timed_out, hung=hung,
               steps_completed=a["steps_completed"], target_steps=steps,
               effective_backend=a["effective_backend"], gloo_fallback=a["gloo_fallback"],
               xccl_rank_to_gpu=a["xccl_rank_to_gpu"],
               xccl_distinct_devices=a["xccl_distinct_devices"],
               devices_with_activity=a["devices_with_activity"],
               peak_xpu_mem_mib=a["peak_xpu_mem_mib"],
               first_sync_s=a["first_sync_s"], second_sync_s=a["second_sync_s"],
               broadcasts=len(a["broadcast_durations_s"]),
               weight_probe_active=a["weight_probe_active"],
               actor_params_changed=a["actor_params_changed"],
               param_equality_matched=a["param_equality_matched"],
               param_equality_mismatched=a["param_equality_mismatched"],
               sleep_markers=a["sleep_markers"],
               vllm_sleep_downgraded=a["vllm_sleep_downgraded"],
               fatal_signatures=a["fatal_signatures"],
               kernel_events=[l for l in dmesg_delta.splitlines()
                              if re.search(r"Engine reset|GPU HANG|timed out", l)],
               criteria=crit, cleanup_ok=cleanup_ok, launched=True,
               suite_abort=(not cleanup_ok), ts=now_iso(), **timings)
    return rec


# --------------------------------------------------------------------------------------
# Suite driver
# --------------------------------------------------------------------------------------
def capture_environment(env_dir: Path):
    env_dir.mkdir(parents=True, exist_ok=True)
    probe = (
        "import json,torch,sys\n"
        "d={'python':sys.version.split()[0],'torch':torch.__version__,"
        "'xpu_available':torch.xpu.is_available(),'xpu_count':torch.xpu.device_count()}\n"
        "import torch.distributed as dist\n"
        "d['xccl_available']=bool(getattr(dist,'is_xccl_available',lambda:False)())\n"
        "for m in ('vllm','deepspeed','ray','transformers'):\n"
        "    try:\n"
        "        d[m]=__import__(m).__version__\n"
        "    except Exception as e:\n"
        "        d[m]='ERR '+type(e).__name__\n"
        "d['xpu_names']=[torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]\n"
        "print(json.dumps(d,indent=2))\n"
    )
    # Must run under child_env(): without the venv lib on LD_LIBRARY_PATH the base miniforge
    # libur_loader wins and `import torch` dies with
    # "libsycl.so.9: undefined symbol: urDeviceWaitExp". sh() does not pass an env, so the
    # XCCL suite recorded an ImportError traceback instead of the version manifest.
    try:
        p = subprocess.run([str(PYTHON), "-c", probe], capture_output=True, text=True,
                           timeout=180, env=child_env())
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        out = "<version probe timed out>"
    (env_dir / "versions.json").write_text(out)

    rc, git = sh(f"git -C {shlex.quote(str(REPO))} log -1 --format='%H%n%d%n%s' && "
                 f"git -C {shlex.quote(str(REPO))} status --short", timeout=60)
    (env_dir / "openrlhf_revision.txt").write_text(git)
    rc, disc = sh("timeout 25 xpu-smi discovery", timeout=40)
    (env_dir / "xpu_discovery.txt").write_text(disc)
    (env_dir / "dmesg_baseline.txt").write_text(dmesg_tail())

    # checksums of the code actually under test
    csum = []
    for f in sorted(REPO.rglob("openrlhf/**/*.py")):
        try:
            csum.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(REPO)}")
        except Exception:
            pass
    (env_dir / "openrlhf_sources.sha256").write_text("\n".join(csum) + "\n")
    (env_dir / "suite_script.sha256").write_text(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + "  run_gloo_matrix.py\n")
    return out


def write_summary(run_dir: Path, records: list, planned: list, started: str):
    tally = {}
    for r in records:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    done = len(records)
    total = len(planned)

    L = ["# XCCL weight-sync validation suite", "",
         f"- Run id: `{run_dir.name}`",
         f"- Started: {started}",
         f"- Updated: {now_iso()}",
         f"- Backend: GLOO only (no xccl cases)",
         f"- Physical devices: 2x Intel Arc Pro B70",
         f"- Steps per case: {N_STEPS}",
         f"- Executable cases: {total} (20 scenarios x 4 DeepSpeed stages)",
         f"- Declared invalid, not run: X5-X8 x 4 stages = 16 cases",
         f"- Progress: **{done}/{total}**", ""]
    L.append("## Tally")
    L.append("")
    L.append("| Status | Count |")
    L.append("|---|---|")
    for k in ("PASS", "FAIL", "BLOCKED", "UNSCHEDULABLE"):
        L.append(f"| {k} | {tally.get(k, 0)} |")
    L.append(f"| **executed** | **{done}** |")
    L.append("")

    L.append("## Results by case")
    L.append("")
    L.append("| Case | Actor WS | Critic WS | vLLM | vSleep | dsSleep | Stage | Status | Steps | Devices | 1st sync (s) | Failure phase |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    by_id = {r["case_id"]: r for r in records}
    for sc, stage in planned:
        cid = f"{sc['id']}-Z{stage}"
        r = by_id.get(cid)
        st = r["status"] if r else "pending"
        steps = f"{r['steps_completed']}/{N_STEPS}" if r and r.get("launched") else ("-" if r else "")
        devs = r.get("xccl_distinct_devices") or r.get("devices_with_activity") or "" if r else ""
        fs = r.get("first_sync_s") if r else ""
        ph = (r.get("failure_phase") or "") if r and r["status"] != "PASS" else ""
        L.append(f"| {cid} | {sc['actor_ws']} | {sc['critic_ws']} | "
                 f"{sc['engines']}xTP{sc['tp']} | {'On' if sc['vllm_sleep'] else 'Off'} | "
                 f"{'On' if sc['ds_sleep'] else 'Off'} | "
                 f"{'0' if stage == 0 else f'Z{stage}'} | **{st}** | {steps} | {devs} | "
                 f"{fs if fs is not None else ''} | {ph[:70]} |")
    L.append("")

    L.append("## Declared invalid (not executed)")
    L.append("")
    for sid in ("X5", "X6", "X7", "X8"):
        L.append(f"### {sid} (all 4 stages) — INVALID")
        L.append("")
        L.append(BY_ID[sid]["invalid_reason"])
        L.append("")

    fails = [r for r in records if r["status"] != "PASS"]
    if fails:
        L.append("## Failure detail")
        L.append("")
        for r in fails:
            L.append(f"### {r['case_id']} — {r['status']}")
            L.append("")
            L.append(f"- Phase: {r.get('failure_phase')}")
            L.append(f"- Error: {r.get('error')}")
            if r.get("fatal_signatures"):
                L.append(f"- Signatures: {r['fatal_signatures']}")
            if r.get("kernel_events"):
                L.append(f"- Kernel events (dmesg): {r['kernel_events']}")
            L.append(f"- Steps reached: {r.get('steps_completed')}/{N_STEPS}")
            L.append(f"- Log: `cases/{r['case_id']}/train.log`")
            L.append("")
    (run_dir / "SUITE_SUMMARY.md").write_text("\n".join(L) + "\n")


def parse_only(spec: str):
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and tok.upper().startswith("X"):
            a, b = tok.upper().split("-")
            for i in range(int(a[1:]), int(b.lstrip("X")) + 1):
                ids.append(f"X{i}")
        else:
            ids.append(tok.upper())
    return ids


def main():
    global MEMORY_LEAN, LORA_RANK
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="scenario ids, e.g. X9,X10 or X9-X12")
    ap.add_argument("--stages", default="0,1,2,3", help="DeepSpeed stages to run")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--gpu-mem-util", type=float, default=GPU_MEM_UTIL)
    ap.add_argument("--case-timeout", type=int, default=CASE_TIMEOUT)
    ap.add_argument("--resume", help="existing run id under results/ to continue")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--include-invalid", action="store_true",
                    help="also attempt X5-X8 (expected UNSCHEDULABLE)")
    ap.add_argument("--continue-on-unhealthy", action="store_true",
                    help="do not abort the suite when the health gate fails")
    ap.add_argument("--extra-arg", action="append", default=[], metavar="ARG",
                    help="DIAGNOSTIC ONLY: append a raw arg to every launched command, "
                         "e.g. --extra-arg --ds.adam_offload. Recorded in the report.")
    ap.add_argument("--set-env", action="append", default=[], metavar="K=V",
                    help="DIAGNOSTIC ONLY: override a child env var")
    ap.add_argument("--unset-env", action="append", default=[], metavar="K",
                    help="DIAGNOSTIC ONLY: remove a child env var")
    ap.add_argument("--tag", default="", help="suffix appended to the run id, e.g. --tag abtest")
    ap.add_argument("--venv", default=str(VENV),
                    help="torch-2.13 venv to run in (default: openrlhf-xccl-auto-detect-213, "
                         "the only one whose editable vLLM install still resolves)")
    ap.add_argument("--no-memory-lean", dest="memory_lean", action="store_false", default=True,
                    help="disable the memory-lean profile (LoRA + gradient checkpointing + "
                         "shorter sequences). Restores full fine-tuning at max_len 512 / "
                         "micro_batch 4, which OOMs the heavier colocated cases on this box.")
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK,
                    help=f"LoRA rank for actor AND critic (default {LORA_RANK}; 0 = no LoRA). "
                         "Lower = less optimizer/gradient state.")
    args = ap.parse_args()

    set_venv(args.venv)
    MEMORY_LEAN = args.memory_lean
    LORA_RANK = args.lora_rank
    EXTRA_ARGS.extend(args.extra_arg)
    for kv in args.set_env:
        k, _, v = kv.partition("=")
        ENV_OVERRIDES[k] = v
    for k in args.unset_env:
        ENV_OVERRIDES[k] = None

    if args.list:
        print(f"{'ID':5} {'aWS':>3} {'cWS':>3} {'vLLM':>7} {'vSleep':>7} {'dsSleep':>8}  mode        valid")
        for s in SCENARIOS:
            print(f"{s['id']:5} {s['actor_ws']:>3} {s['critic_ws']:>3} "
                  f"{str(s['engines']) + 'xTP' + str(s['tp']):>7} "
                  f"{'On' if s['vllm_sleep'] else 'Off':>7} "
                  f"{'On' if s['ds_sleep'] else 'Off':>8}  {s['mode']:<11} "
                  f"{'yes' if s['valid'] else 'NO'}")
        valid = [s for s in SCENARIOS if s["valid"]]
        print(f"\nvalid scenarios: {len(valid)}  x 4 stages = {len(valid) * 4} executable cases")
        print(f"invalid: {[s['id'] for s in SCENARIOS if not s['valid']]}")
        return 0

    stages = [int(x) for x in args.stages.split(",") if x.strip() != ""]
    wanted = parse_only(args.only) if args.only else None
    scen = [s for s in SCENARIOS
            if (s["valid"] or args.include_invalid)
            and (wanted is None or s["id"] in wanted)]
    planned = [(s, st) for s in scen for st in stages]

    if args.dry_run:
        for s, st in planned:
            argv, r = build_case(s, st, CASE_GPU_MEM_UTIL.get(f"{s['id']}-Z{st}", args.gpu_mem_util), args.steps)
            print(f"\n===== {s['id']}-Z{st} =====")
            if r.get("blocking_conflict"):
                print("NOT LAUNCHABLE: " + r["blocking_conflict"])
            print(" \\\n    ".join(shlex.quote(x) for x in argv))
        print(f"\n{len(planned)} cases planned.")
        return 0

    results_root = SUITE_DIR / "results"
    if args.resume:
        run_dir = results_root / args.resume
        if not run_dir.is_dir():
            print(f"no such run: {run_dir}", file=sys.stderr)
            return 2
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        run_dir = results_root / (datetime.now().strftime("run_%Y%m%d_%H%M%S") + suffix)
    (run_dir / "cases").mkdir(parents=True, exist_ok=True)

    jsonl = run_dir / "results.jsonl"
    records = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    done_ids = {r["case_id"] for r in records}

    started = now_iso()
    print("=" * 78)
    print(f"XCCL weight-sync matrix suite   run_dir={run_dir}")
    print(f"repo={REPO}\nvenv={VENV}")
    print("=" * 78)
    envjson = capture_environment(run_dir / "environment")
    print(envjson.strip())
    # NOTE: deliberately NOT calling set_wedged_mode() here. Writing xe's wedged_mode debugfs
    # node makes the driver push a GuC ADS scheduler-policy update, and "failed to enable GuC
    # scheduling policies: -ETIME" is exactly the first error in the 2026-09-07 GuC hang. On a
    # healthy card there is nothing to un-wedge, so this is risk with no benefit; recover_xpus()
    # still sets it after a PCI re-probe, where a wedge is the thing we are clearing.
    print("-" * 78)
    todo = [(s, st) for s, st in planned if f"{s['id']}-Z{st}" not in done_ids]
    print(f"planned={len(planned)}  already done={len(planned) - len(todo)}  to run={len(todo)}")
    print(f"steps/case={args.steps}  gpu_mem_util={args.gpu_mem_util}  "
          f"case_timeout={args.case_timeout}s")
    print("-" * 78)

    progress = (run_dir / "progress.log").open("a")

    def tally():
        t = {}
        for r in records:
            t[r["status"]] = t.get(r["status"], 0) + 1
        return t

    for i, (sc, stage) in enumerate(planned, 1):
        cid = f"{sc['id']}-Z{stage}"
        if cid in done_ids:
            continue
        print(f"\n[{i}/{len(planned)}] {cid}  ({now_iso()})")
        gmu = CASE_GPU_MEM_UTIL.get(cid, args.gpu_mem_util)
        if gmu != args.gpu_mem_util:
            print(f"      gpu_memory_utilization={gmu} (per-case override; default "
                  f"{args.gpu_mem_util} sizes this cell's KV cache negative)")
        rec = run_case(sc, stage, run_dir, gmu, args.steps, args.case_timeout)
        records.append(rec)
        with jsonl.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

        t = tally()
        done = len(records)
        line = (f"[{done}/{len(planned)}] {cid} -> {rec['status']}"
                f"  steps={rec.get('steps_completed', '-')}/{args.steps}"
                f"  {rec.get('total_s', '?')}s"
                f"  || PASS={t.get('PASS', 0)} FAIL={t.get('FAIL', 0)} "
                f"BLOCKED={t.get('BLOCKED', 0)} UNSCHEDULABLE={t.get('UNSCHEDULABLE', 0)}")
        print("      " + line)
        if rec["status"] != "PASS":
            print(f"      reason: {str(rec.get('error'))[:300]}")
        progress.write(line + "\n")
        progress.flush()
        write_summary(run_dir, records, planned, started)

        if rec.get("suite_abort") and not args.continue_on_unhealthy:
            print("\n!! environment unhealthy or cleanup failed -- stopping the suite.")
            print(f"!! reason: {rec.get('error')}")
            break

    write_summary(run_dir, records, planned, started)
    t = tally()
    print("\n" + "=" * 78)
    print(f"FINAL  PASS={t.get('PASS', 0)}  FAIL={t.get('FAIL', 0)}  "
          f"BLOCKED={t.get('BLOCKED', 0)}  UNSCHEDULABLE={t.get('UNSCHEDULABLE', 0)}  "
          f"of {len(planned)} planned")
    print(f"summary: {run_dir / 'SUITE_SUMMARY.md'}")
    print("=" * 78)
    progress.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
