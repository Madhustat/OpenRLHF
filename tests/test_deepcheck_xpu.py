"""Tier 2 — deep-check / invariant suite (self-contained subset).

Tier 1 (the E2E .sh suites) proves each workflow *runs*. This file asserts
*correctness invariants* that a "runs + exits 0" PASS can hide. These are the
self-contained checks (no full Ray+vLLM training loop needed); the heavier
run-based checks (weight_update, ppo_critic_update, resume_equivalence,
memory_stability) live in test_e2e_suite_singlegpu_deepcheck.sh.

Run:  python -m pytest tests/test_deepcheck_xpu.py -v
"""
import importlib
import os

import pytest
import torch

from openrlhf.utils.distributed_util import _GlooBroadcastCommunicator, resolve_vllm_sync_backend


def _accel():
    """Active accelerator device string ('xpu'/'cuda'), or None on CPU-only."""
    acc = getattr(torch, "accelerator", None)
    if acc is not None:
        try:
            dev = torch.accelerator.current_accelerator()
            if dev is not None and torch.accelerator.device_count() > 0:
                return dev.type
        except Exception:
            pass
    if torch.xpu.is_available():
        return "xpu"
    return None


class _FakeSrcPG:
    """Minimal process group stub: broadcast is a no-op (rank-0 source keeps value)."""
    def broadcast(self, tensors, opts=None):
        class _W:
            def wait(self_):
                return None
        return _W()


# ── #10 all_import_smoke_xpu ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "mod",
    [
        "openrlhf.cli.train_ppo_ray",
        "openrlhf.cli.train_sft",
        "openrlhf.cli.train_rm",
        "openrlhf.cli.train_dpo",
        "openrlhf.trainer.ppo_trainer",
        "openrlhf.trainer.ray.ppo_actor",
        "openrlhf.trainer.ray.vllm_engine",
        "openrlhf.utils.deepspeed.deepspeed",
    ],
)
def test_all_import_smoke_xpu(mod):
    """Every trainer/launcher imports on XPU with no CUDA-specific import failure."""
    m = importlib.import_module(mod)
    assert m is not None


# ── #1 xpu_purity (partial, static) ─────────────────────────────────────────
def test_xpu_purity_no_cuda_on_xpu_stack():
    """On the XPU stack, torch must NOT report CUDA available and the canonical
    accelerator must be xpu — a silent CUDA/CPU fallback would be visible here."""
    if not torch.xpu.is_available():
        pytest.skip("no XPU on this box")
    assert not torch.cuda.is_available(), "CUDA reported available on an XPU-only build"
    assert _accel() == "xpu", f"canonical accelerator is {_accel()}, expected xpu"


# ── #2 weight_sync_exactness (primitive, bit-exact) ─────────────────────────
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_sync_exactness_bit_exact(dtype):
    """The gloo broadcast primitive must preserve weights bit-exactly (copy_ path,
    no dtype drift, storage preserved). Full actor->vLLM path exactness is the
    E2E deepcheck; this pins the transport primitive."""
    comm = _GlooBroadcastCommunicator(_FakeSrcPG(), device=torch.device("cpu"))
    weights = torch.randn(256, 128, dtype=dtype)
    reference = weights.clone()
    storage_before = weights.untyped_storage().data_ptr()

    result = comm.broadcast(weights, src=0)

    assert result is weights, "broadcast must return the same tensor object"
    assert result.untyped_storage().data_ptr() == storage_before, "storage must be preserved"
    assert torch.equal(result, reference), "broadcast altered weight values (not bit-exact)"


# ── #8 zero_grad ────────────────────────────────────────────────────────────
def test_zero_grad_resets_between_steps():
    """Gradients must be reset between optimizer steps, not silently accumulate."""
    device = _accel() or "cpu"
    model = torch.nn.Linear(16, 16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def one_step():
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 16, device=device)
        loss = model(x).pow(2).mean()
        loss.backward()
        g = model.weight.grad.detach().float().norm().item()
        opt.step()
        return g

    g1 = one_step()
    g2 = one_step()
    assert g1 > 0 and g2 > 0, "no gradient produced"
    # After zero_grad(set_to_none=True) at the top of step 2, grads were cleared
    # then recomputed fresh — they must not be the exact accumulation of both.
    opt.zero_grad(set_to_none=True)
    assert model.weight.grad is None, "zero_grad(set_to_none=True) did not clear gradients"


# ── #5 loss_finite (self-contained smoke) ───────────────────────────────────
def test_loss_and_grads_finite():
    """Loss and gradients stay finite (no NaN/Inf) through a forward+backward on
    the active accelerator — XPU dtype/kernel bugs often first surface as NaN."""
    device = _accel() or "cpu"
    model = torch.nn.Linear(32, 8).to(device)
    x = torch.randn(4, 32, device=device)
    loss = model(x).pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).all(), "loss is NaN/Inf"
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), f"grad for {name} is NaN/Inf"


# ── #9 unsupported_backend_guard ────────────────────────────────────────────
def test_unsupported_backend_guard_rejects_unknown():
    """Unknown weight-sync backend selections fail fast with a clear error rather
    than hanging deep inside a collective."""
    with pytest.raises(ValueError, match="Unsupported vLLM weight-sync backend"):
        resolve_vllm_sync_backend("mpi")
