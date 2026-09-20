"""Make the OpenRLHF weight-freshness probe actually reach stdout.

The probes in the repo (ppo_actor.py broadcast_to_vllm, vllm_worker_wrap.py
update_weight) log through ``logging.getLogger("weight_freshness").info(...)``. That logger
has no handler and no level of its own, so it inherits the root logger, which in a Ray
worker sits at WARNING -- every probe record is dropped and the checksums never appear in
the captured log. Pass criteria 5/8/10 (actor params changed, vLLM params updated,
actor==vLLM after sync) then cannot be verified at all.

Fixing that in the repo is out of scope (the repo must not be modified), so this
sitecustomize -- injected only via PYTHONPATH by run_xccl_matrix.py -- gives that one
logger its own INFO-level stdout handler. Nothing else is touched: the root logger's level
is left alone, so Ray/vLLM/DeepSpeed verbosity is unchanged.

sitecustomize is imported automatically at interpreter startup for every process that
inherits PYTHONPATH, which is exactly what is needed: the probes run inside Ray worker
processes, not in the driver.

Second job, added 2026-09-08: optionally force DeepSpeed's OWN process-group backend to
gloo. See the module-level comment block on _force_ds_backend() below for why.

Third job, added 2026-09-09: optionally mirror RANK/WORLD_SIZE into the PMI_RANK/PMI_SIZE
variables oneCCL's ATL/MPI transport needs. See the comment block below _RANK_MIRROR for why.
"""

import importlib.util
import logging
import os
import sys

_probe_logger = logging.getLogger("weight_freshness")
if not _probe_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _probe_logger.addHandler(_handler)
    # propagate=False so the record is emitted exactly once even if some later
    # basicConfig() attaches a root handler.
    _probe_logger.propagate = False
_probe_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Force DeepSpeed's own process-group backend (opt-in via OPENRLHF_DS_PG_BACKEND)
# ---------------------------------------------------------------------------
# WHY: "gloo weight sync" only covers the actor -> vLLM weight broadcast. DeepSpeed's own
# torch.distributed process group -- the one that all-reduces GRADIENTS across actor ranks --
# is chosen independently, by XPU_Accelerator.communication_backend_name(), which hard-codes
# 'ccl'/'xccl' with no config or env override (xpu_accelerator.py:24-27). So at actor
# world_size 2 the gradient reduce runs over XCCL no matter what --vllm.sync_backend says.
#
# MEASURED 2026-09-08, gloo matrix run full80: every actor_ws=1 case that launched trained
# 5/5 steps, while every actor_ws=2 case (X9-Z1/Z2/Z3, X10-Z0/Z1/Z2) died with
# `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`
# raised from the first micro-batch's backward. Ruled out as causes:
#   * memory -- X9-Z1 peaked at 12.2/10.2 GiB of 32.6, while the PASSING X4-Z2 peaked at 29.9;
#   * hardware -- `journalctl -k` recorded ZERO CAT errors / resets / GuC timeouts for the
#     whole run, and both cards stayed `Device State: normal`;
#   * a ZeRO-stage bucketing quirk -- stage 0 (X10-Z0) fails identically to stage 2.
# The only variable is actor world size, i.e. whether that XCCL gradient collective runs.
#
# So this moves DeepSpeed's PG to gloo, which is the same workaround already proven for the
# two-live-XCCL-communicators segfault. Gradients then stage through CPU.
#
# WHAT THIS COSTS, stated plainly: for actor_ws=2 cases the gradient all-reduce is no longer
# XCCL, so those cases do NOT validate XCCL gradient reduction -- they validate scheduling,
# placement, sleep/wake and gloo weight sync, which is what this matrix exists to measure.
# Gradients over CPU are slower, but the model is 0.5B and each case runs 5 steps.
# Leave OPENRLHF_DS_PG_BACKEND unset to get the stock XCCL behaviour back.
_DS_PG_BACKEND = os.environ.get("OPENRLHF_DS_PG_BACKEND", "").strip()
_DS_ACCEL_MODULE = "deepspeed.accelerator.xpu_accelerator"

if _DS_PG_BACKEND:

    class _DsBackendPatcher:
        """Patch XPU_Accelerator.communication_backend_name right after its module loads.

        A meta_path finder is used rather than patching at startup because sitecustomize runs
        long before DeepSpeed is importable, and importing DeepSpeed here would both be slow
        and change import ordering inside every Ray worker.
        """

        _reentrant = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _DS_ACCEL_MODULE or _DsBackendPatcher._reentrant:
                return None
            # Guard against infinite recursion: find_spec() below re-enters sys.meta_path.
            _DsBackendPatcher._reentrant = True
            try:
                spec = importlib.util.find_spec(fullname)
            except Exception:
                return None
            finally:
                _DsBackendPatcher._reentrant = False
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return None
            _orig_exec = spec.loader.exec_module

            def exec_module(module):
                _orig_exec(module)
                accel = getattr(module, "XPU_Accelerator", None)
                if accel is not None:
                    accel.communication_backend_name = lambda self: _DS_PG_BACKEND
                    print(f"[sitecustomize] DeepSpeed XPU comm backend forced to "
                          f"'{_DS_PG_BACKEND}' (pid={os.getpid()})", flush=True)

            # The spec's loader is a fresh SourceFileLoader for this module, so rebinding
            # exec_module on it affects only this import.
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _DsBackendPatcher())


# ---------------------------------------------------------------------------
# Mirror RANK/WORLD_SIZE into PMI_RANK/PMI_SIZE (opt-in via OPENRLHF_CCL_MPI_RANK_SHIM)
# ---------------------------------------------------------------------------
# WHY: intel/torch-xpu-ops#4238 and #4241 (2x Battlemage/PCIe, no XeLink -- same class of
# hardware as this box) both document CCL_ATL_TRANSPORT=mpi as part of a workaround for
# collective failures on this exact topology. Tried directly on X12-Z3 on 2026-09-09: it fails
# immediately, before any real collective, with
#   Abort(...) Fatal error in internal_Group_incl: ... Duplicate ranks in rank array at index 1,
#   has value 0 which is also the value at index 0
# because oneCCL's ATL/MPI layer discovers rank via PMI_RANK/PMI_SIZE (confirmed present as
# literal strings in libccl.so.1), which only exist when the process is launched under `mpirun`
# -- OpenRLHF launches actors as Ray workers, not via mpirun, so PMI_RANK is never set and every
# rank's ATL layer falls back to the same default (0).
#
# RANK/WORLD_SIZE ARE set correctly per worker, but only inside launcher.py's _setup_distributed
# (openrlhf/trainer/ray/launcher.py:28-31), at Python runtime -- long after sitecustomize has
# already run at interpreter startup, so this can't be solved by reading os.environ once here.
# Instead this hooks os.environ.__setitem__ itself so that the moment launcher.py sets RANK
# (whatever order it does so in), PMI_RANK/PMI_SIZE (what the "did not find MPI-launcher
# specific variables" warning is looking for) and CCL_LOCAL_RANK/CCL_LOCAL_SIZE (oneCCL's own
# direct override, confirmed present as literal strings in libccl.so.1) all get mirrored
# immediately, before any oneCCL/DeepSpeed process-group init that happens afterward.
_MPI_RANK_SHIM = os.environ.get("OPENRLHF_CCL_MPI_RANK_SHIM", "") == "1"

if _MPI_RANK_SHIM:
    import builtins as _builtins

    _orig_setitem = os.environ.__class__.__setitem__

    def _mirroring_setitem(self, key, value):
        _orig_setitem(self, key, value)
        if key == "RANK":
            _orig_setitem(self, "PMI_RANK", value)
            _orig_setitem(self, "CCL_LOCAL_RANK", value)
            print(f"[sitecustomize] mirrored RANK={value} -> PMI_RANK, CCL_LOCAL_RANK "
                  f"(pid={os.getpid()})", flush=True)
        elif key == "WORLD_SIZE":
            _orig_setitem(self, "PMI_SIZE", value)
            _orig_setitem(self, "CCL_LOCAL_SIZE", value)
            print(f"[sitecustomize] mirrored WORLD_SIZE={value} -> PMI_SIZE, CCL_LOCAL_SIZE "
                  f"(pid={os.getpid()})", flush=True)

    os.environ.__class__.__setitem__ = _mirroring_setitem
