"""Regression test for the ZeRO-3 overlap_comm silent-default gap.

Background (2026-09 investigation): `get_train_ds_config(overlap_comm=False, stage=3, ...)` used
to omit the `overlap_comm` key from the generated `zero_optimization` dict whenever `overlap_comm`
was falsy, instead of writing `False` explicitly. DeepSpeed's own ZeRO config resolves an absent
`overlap_comm` to `self.stage == ZeroStageEnum.weights` (i.e. `True` for stage 3) --
so OpenRLHF's own default of `False` never actually reached the live config for ZeRO-3: every
stage-3 run silently got `overlap_comm=True` regardless of the `--ds.overlap_comm` flag.

This mattered: a minimal DeepSpeed ZeRO-3 reproducer with `overlap_comm=True` (the silent default)
crashed deterministically with a native SIGSEGV on 2x Intel Battlemage B70 (PCIe, no XeLink) under
the ATL/OFI oneCCL transport (`ccl_worker_func -> ... -> urEventGetInfo ->
libze_intel_gpu.so.1`). The identical reproducer with only `overlap_comm=False` added passed
3/3 runs, 300 combined steps. The same fix, applied to this function, was then validated on the
real X12-Z3 OpenRLHF+Ray+vLLM+Gloo+Critic case: 3/3 clean 5-step passes, zero failures -- the first
successful full-stack run in that entire investigation.

This test does not require XPU hardware or DeepSpeed's XPU accelerator; it only inspects the
config dict `get_train_ds_config` produces.
"""

from openrlhf.utils.deepspeed.deepspeed_utils import get_train_ds_config


def test_stage3_overlap_comm_false_is_written_explicitly():
    """The whole point of the fix: omitting the key at stage 3 means DeepSpeed's own resolver
    silently substitutes True. Writing `False` is the only way OpenRLHF's own default reaches
    the live config.
    """
    config = get_train_ds_config(offload=False, stage=3, overlap_comm=False)
    zero_opt = config["zero_optimization"]

    assert "overlap_comm" in zero_opt, (
        "overlap_comm key is missing from the stage-3 config -- DeepSpeed's own ZeroStageEnum. "
        "weights default will silently resolve this to True (see module docstring)"
    )
    assert zero_opt["overlap_comm"] is False


def test_stage3_overlap_comm_true_still_sets_contiguous_gradients():
    """Make sure the fix didn't disturb the existing True-branch behavior."""
    config = get_train_ds_config(offload=False, stage=3, overlap_comm=True)
    zero_opt = config["zero_optimization"]

    assert zero_opt["overlap_comm"] is True
    assert zero_opt["contiguous_gradients"] is True


def test_stage2_overlap_comm_false_does_not_force_a_key():
    """The fix is scoped to stage 3 only -- stage 1/2 keep the original omit-if-false behavior,
    since DeepSpeed's dynamic default there is not known to have the same silent-True trap
    (ZeroStageEnum.weights is stage 3 specifically).
    """
    config = get_train_ds_config(offload=False, stage=2, overlap_comm=False)
    zero_opt = config["zero_optimization"]

    assert "overlap_comm" not in zero_opt
