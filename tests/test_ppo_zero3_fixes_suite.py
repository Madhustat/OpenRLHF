"""Consolidated regression suite for the three PPO / ZeRO-3 fixes found in the 2026-09
XPU DeepSpeed-sleep investigation (2x Intel Arc Pro B70, Battlemage).

Run everything in this file with one command:

    pytest tests/test_ppo_zero3_fixes_suite.py -v

or directly:

    python tests/test_ppo_zero3_fixes_suite.py

None of these tests require GPU/XPU hardware, DeepSpeed's accelerator runtime, or a live Ray
cluster -- they check the plain-Python config-building logic and the trainer's source code
shape, which is what each fix actually changed.

Sections below, each corresponding to one commit:

  1. FIX 1 -- ppo_actor.py: weight-freshness-probe collective-order deadlock (commit 3dad9b30)
     Production-relevance: none by default. Only matters when the pre-existing
     OPENRLHF_WEIGHT_PROBE=1 diagnostic instrumentation is turned on (off by default) --
     this investigation relied on that probe to verify actor/vLLM weight checksums, so the
     fix was required for our own test methodology, not for stock production training.

  2. FIX 2 -- deepspeed_utils.py: overlap_comm silent stage-3 default (commit c11455bb)
     Production-relevance: real, for every backend. Every ZeRO-3 run was silently getting
     overlap_comm=True regardless of the CLI's own default of False.

  3. FIX 3 -- deepspeed_utils.py: FusedAdam-only sleep-offload assertion (commit 2fcd8d7b)
     Production-relevance: real, for any box where `icpx` can't JIT-build FusedAdam (this
     box) and --ds.enable_sleep is used with ZeRO-3.
"""

import ast
import inspect
import textwrap
import types

import pytest


# ============================================================================================
# FIX 1 -- ppo_actor.py: weight-freshness-probe collective-order deadlock (commit 3dad9b30)
# ============================================================================================
#
# Background: ppo_actor.py's OPENRLHF_WEIGHT_PROBE block used to gate its ENTIRE body --
# including a real `deepspeed.zero.GatheredParameters` call -- on
# `torch.distributed.get_rank() == 0`. Under ZeRO-3 (GatheredParameters is a no-op below
# stage 3) with actor world_size > 1 (a lone rank can't desync from itself),
# GatheredParameters.__enter__/__exit__ issue a REAL collective on DeepSpeed's own process
# group, matched by call order, not by which parameter is being gathered. Rank 0 running
# extra probe-only collective calls before every other rank permanently shifted it out of
# phase with them for the rest of the process's lifetime, so the very next
# GatheredParameters use (the real broadcast loop) paired rank 0's call for one parameter
# with another rank's call for a different parameter -- a permanent deadlock with no error.
# Proven live with py-spy: rank 0 stuck in the probe's GatheredParameters while another rank
# was already several parameters into the main sync loop.
#
# Fix: every rank enters the identical GatheredParameters call, for the same parameter, in
# the same order (the parameter-selection loop is already deterministic and
# rank-independent) -- only the checksum read and the log line stay rank-0-only. These tests
# prove that shape statically, from the actual source of `broadcast_to_vllm`.

from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer


def _probe_block_source():
    """Return the OPENRLHF_WEIGHT_PROBE if-block's AST node and its test-expression source."""
    src = inspect.getsource(ActorPPOTrainer.broadcast_to_vllm)
    tree = ast.parse(textwrap.dedent(src))
    func = tree.body[0]
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if "OPENRLHF_WEIGHT_PROBE" in test_src:
                return node, test_src
    raise AssertionError("OPENRLHF_WEIGHT_PROBE block not found in broadcast_to_vllm -- probe removed?")


def test_probe_outer_gate_does_not_check_rank():
    """The if-statement gating the whole probe block must NOT also require rank == 0.

    This is the exact bug: `if ... OPENRLHF_WEIGHT_PROBE ... and torch.distributed.get_rank()
    == 0:` wrapped the entire block, including the GatheredParameters collective. If this
    regresses, only rank 0 re-enters the probe's collective and the deadlock returns.
    """
    _, test_src = _probe_block_source()
    assert "get_rank" not in test_src, (
        f"probe's outer gate checks rank ({test_src!r}) -- this reintroduces the "
        "rank-asymmetric GatheredParameters deadlock (see section docstring above)"
    )


def test_gathered_parameters_call_is_not_rank_gated():
    """Every `with ...GatheredParameters(...)` inside the probe block must be unconditional
    w.r.t. rank -- i.e. not nested inside its own `if rank == 0` (the same bug one level
    deeper). The checksum computation *inside* the gather is allowed to be rank-gated; only
    entering/exiting the collective itself must be symmetric.
    """
    node, _ = _probe_block_source()
    gathered_with_nodes = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.With)
        and any("GatheredParameters" in ast.unparse(item.context_expr) for item in n.items)
    ]
    assert gathered_with_nodes, "no GatheredParameters call found inside the probe block"

    for with_node in gathered_with_nodes:
        for n in ast.walk(node):
            if isinstance(n, ast.If) and "get_rank" in ast.unparse(n.test):
                assert with_node not in ast.walk(n) or with_node not in n.body, (
                    "found a GatheredParameters call nested inside a rank-gated `if` -- this "
                    "is the same class of collective-order bug as the original deadlock"
                )


def test_checksum_read_is_still_rank_gated():
    """Sanity check the fix didn't overcorrect: only rank 0 should compute/log the checksum
    (that part was never required to be symmetric -- only the collective entry was).
    """
    src = inspect.getsource(ActorPPOTrainer.broadcast_to_vllm)
    assert "_is_probe_rank0" in src or "get_rank() == 0" in src, (
        "expected some rank-0-only gate to remain for the checksum/log -- if this is gone "
        "the probe now logs from every rank"
    )


# ============================================================================================
# FIX 2 -- deepspeed_utils.py: overlap_comm silent stage-3 default (commit c11455bb)
# ============================================================================================
#
# Background: `get_train_ds_config(overlap_comm=False, stage=3, ...)` used to omit the
# `overlap_comm` key from the generated `zero_optimization` dict whenever `overlap_comm` was
# falsy, instead of writing `False` explicitly. DeepSpeed's own ZeRO config resolves an
# absent `overlap_comm` to `self.stage == ZeroStageEnum.weights` (True for stage 3) -- so
# OpenRLHF's own default of False never actually reached the live config for ZeRO-3: every
# stage-3 run silently got overlap_comm=True regardless of the --ds.overlap_comm flag.
#
# This mattered: a minimal DeepSpeed ZeRO-3 reproducer with overlap_comm=True (the silent
# default) crashed deterministically with a native SIGSEGV on 2x Intel Battlemage B70
# (PCIe, no XeLink) under the ATL/OFI oneCCL transport. The identical reproducer with only
# overlap_comm=False added passed 3/3 runs, 300 combined steps. The same fix, applied to
# this function, was then validated on the real X12-Z3 OpenRLHF+Ray+vLLM+Gloo+Critic case:
# 3/3 clean 5-step passes.

from openrlhf.utils.deepspeed.deepspeed_utils import get_train_ds_config


def test_stage3_overlap_comm_false_is_written_explicitly():
    """The whole point of the fix: omitting the key at stage 3 means DeepSpeed's own resolver
    silently substitutes True. Writing `False` is the only way OpenRLHF's own default reaches
    the live config.
    """
    config = get_train_ds_config(offload=False, stage=3, overlap_comm=False)
    zero_opt = config["zero_optimization"]

    assert "overlap_comm" in zero_opt, (
        "overlap_comm key is missing from the stage-3 config -- DeepSpeed's own "
        "ZeroStageEnum.weights default will silently resolve this to True (see section "
        "docstring above)"
    )
    assert zero_opt["overlap_comm"] is False


def test_stage3_overlap_comm_true_still_sets_contiguous_gradients():
    """Make sure the fix didn't disturb the existing True-branch behavior."""
    config = get_train_ds_config(offload=False, stage=3, overlap_comm=True)
    zero_opt = config["zero_optimization"]

    assert zero_opt["overlap_comm"] is True
    assert zero_opt["contiguous_gradients"] is True


def test_stage2_overlap_comm_false_does_not_force_a_key():
    """The fix is scoped to stage 3 only -- stage 1/2 keep the original omit-if-false
    behavior, since DeepSpeed's dynamic default there is not known to have the same
    silent-True trap (ZeroStageEnum.weights is stage 3 specifically).
    """
    config = get_train_ds_config(offload=False, stage=2, overlap_comm=False)
    zero_opt = config["zero_optimization"]

    assert "overlap_comm" not in zero_opt


# ============================================================================================
# FIX 3 -- deepspeed_utils.py: FusedAdam-only sleep-offload assertion (commit 2fcd8d7b)
# ============================================================================================
#
# Background: `--ds.enable_sleep` calls `offload_deepspeed_states()`, which used to
# unconditionally request OffloadStateTypeEnum.optim_states and .hp_params.
# DeepSpeedZeroOptimizer_Stage3.offload_states() hard-asserts
# `self.optimizer.__class__ == deepspeed.ops.adam.fused_adam.FusedAdam` before touching
# either of those two categories (stage3.py:3285) -- both call into FusedAdam-specific
# helpers. The other three categories (lp_params, lp_grads, contiguous_grad_buffer) never
# touch self.optimizer at all -- they only move DeepSpeed's own ZeRO-3 partition buffers --
# so they work with any optimizer. This crashed --ds.enable_sleep outright whenever the
# actor used plain torch.optim.AdamW (forced by OPENRLHF_DS_TORCH_ADAM=1, since building
# FusedAdam needs icpx, unavailable on this box).
#
# Fix: `_optimizer_supports_state_offload()` detects the FusedAdam case and
# `offload_deepspeed_states()` only requests the two FusedAdam-only categories when it
# applies, falling back to the optimizer-independent categories otherwise. Validated on all
# 9 ZeRO-3 + DS-sleep-On topologies in the test matrix (X9/X11/X13/X15/X17/X19/X21/X23):
# all pass 5/5 steps, where they previously hard-crashed on the FusedAdam assertion.

from openrlhf.utils.deepspeed.deepspeed_utils import _optimizer_supports_state_offload


class _FakeFusedAdam:
    pass


class _FakeAdamW:
    pass


def _fake_model(inner_optimizer_class):
    inner = inner_optimizer_class()
    wrapper = types.SimpleNamespace(optimizer=inner)
    return types.SimpleNamespace(optimizer=wrapper)


def test_plain_adamw_is_not_capable():
    import deepspeed.ops.adam.fused_adam  # noqa: F401 -- ensure the real class exists for contrast

    model = _fake_model(_FakeAdamW)
    assert _optimizer_supports_state_offload(model) is False


def test_real_fused_adam_class_is_capable():
    import deepspeed.ops.adam.fused_adam as fused_adam_module

    model = _fake_model(lambda: object.__new__(fused_adam_module.FusedAdam))
    # __new__ without __init__ still has the right __class__ for the identity check.
    assert _optimizer_supports_state_offload(model) is True


def test_missing_inner_optimizer_is_not_capable():
    wrapper = types.SimpleNamespace()  # no .optimizer attribute at all
    model = types.SimpleNamespace(optimizer=wrapper)
    assert _optimizer_supports_state_offload(model) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
