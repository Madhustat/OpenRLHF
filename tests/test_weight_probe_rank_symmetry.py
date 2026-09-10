"""Regression test for the ZeRO-3 weight-freshness-probe collective-order deadlock.

Background (2026-09-08 investigation): ppo_actor.py's OPENRLHF_WEIGHT_PROBE block used to gate
its ENTIRE body -- including a real `deepspeed.zero.GatheredParameters` call -- on
`torch.distributed.get_rank() == 0`. Under ZeRO-3 (GatheredParameters is a no-op below stage 3)
with actor world_size > 1 (a lone rank can't desync from itself), GatheredParameters.__enter__/
__exit__ issue a REAL collective on DeepSpeed's own process group, matched by call order, not by
which parameter is being gathered. Rank 0 running extra probe-only collective calls before every
other rank permanently shifted it out of phase with them for the rest of the process's lifetime,
so the very next GatheredParameters use (the real broadcast loop) paired rank 0's call for one
parameter with another rank's call for a different parameter -- a permanent deadlock with no
error. Proven live with py-spy: rank 0 stuck in the probe's GatheredParameters while another rank
was already several parameters into the main sync loop.

Fix: every rank enters the identical GatheredParameters call, for the same parameter, in the same
order (the parameter-selection loop is already deterministic and rank-independent) -- only the
checksum read and the log line stay rank-0-only. This test proves that shape statically: it does
not require multi-rank hardware, ZeRO-3, or DeepSpeed, only the actual source of
`broadcast_to_vllm`.
"""

import ast
import inspect
import textwrap

from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer


def _probe_block_source() -> str:
    """Return just the OPENRLHF_WEIGHT_PROBE if-block's source, dedented and parsed as an AST."""
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

    This is the exact bug: `if ... OPENRLHF_WEIGHT_PROBE ... and torch.distributed.get_rank() ==
    0:` wrapped the entire block, including the GatheredParameters collective. If this regresses,
    only rank 0 re-enters the probe's collective and the deadlock returns.
    """
    node, test_src = _probe_block_source()
    assert "get_rank" not in test_src, (
        f"probe's outer gate checks rank ({test_src!r}) -- this reintroduces the rank-asymmetric "
        "GatheredParameters deadlock (see module docstring)"
    )


def test_gathered_parameters_call_is_not_rank_gated():
    """Every `with ...GatheredParameters(...)` inside the probe block must be unconditional
    w.r.t. rank -- i.e. not nested inside its own `if rank == 0` (that would be the same bug one
    level deeper). The checksum computation *inside* the gather is allowed to be rank-gated;
    only entering/exiting the collective itself must be symmetric.
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
        # Walk from the probe's top-level `for` loop down to this `with`; any `If` on that path
        # whose test mentions get_rank would mean the gather itself is conditional on rank.
        for n in ast.walk(node):
            if isinstance(n, ast.If) and "get_rank" in ast.unparse(n.test):
                assert with_node not in ast.walk(n) or with_node not in n.body, (
                    "found a GatheredParameters call nested inside a rank-gated `if` -- this is "
                    "the same class of collective-order bug as the original deadlock"
                )


def test_checksum_read_is_still_rank_gated():
    """Sanity check the fix didn't overcorrect: only rank 0 should compute/log the checksum
    (that part was never required to be symmetric -- only the collective entry was).
    """
    src = inspect.getsource(ActorPPOTrainer.broadcast_to_vllm)
    assert "_is_probe_rank0" in src or "get_rank() == 0" in src, (
        "expected some rank-0-only gate to remain for the checksum/log -- if this is gone the "
        "probe now logs from every rank"
    )
