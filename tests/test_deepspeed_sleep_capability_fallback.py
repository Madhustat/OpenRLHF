"""Regression test for the DeepSpeed-sleep FusedAdam capability fallback.

Background (2026-09 investigation): `--ds.enable_sleep` calls `offload_deepspeed_states()`, which
used to unconditionally request `OffloadStateTypeEnum.optim_states` and `.hp_params`.
`DeepSpeedZeroOptimizer_Stage3.offload_states()` hard-asserts
`self.optimizer.__class__ == deepspeed.ops.adam.fused_adam.FusedAdam` before touching either of
those two categories (deepspeed/runtime/zero/stage3.py:3285) -- so ZeRO-3 sleep hard-crashed
whenever the actor used plain `torch.optim.AdamW` (which is what OPENRLHF_DS_TORCH_ADAM=1 forces,
because building FusedAdam requires JIT-compiling it via `icpx`, confirmed absent on this box).

The other three offload categories (`lp_params`, `lp_grads`, `contiguous_grad_buffer`) never touch
`self.optimizer` at all -- they only move DeepSpeed's own ZeRO-3 partition buffers -- so they work
with any optimizer. `_optimizer_supports_state_offload()` detects the FusedAdam case and
`offload_deepspeed_states()` only requests the two FusedAdam-only categories when it applies,
falling back to the optimizer-independent categories otherwise.

Validated on the real X12 topology's ds.enable_sleep variants (X9/X11/X13/X15/X17/X19/X21/X23):
all now pass 5/5 steps where they previously hit the FusedAdam assertion (or, for the
vLLM+DS-sleep-together cases, a related memory-profiling race that this fix also happens to avoid
by keeping more state resident and reducing memory churn).
"""

import types

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
