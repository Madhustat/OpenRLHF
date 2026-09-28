"""Tests for vLLM actor device-visibility configuration (_configure_device_env).

OpenRLHF pins each Ray-assigned accelerator by setting the environment variable that the
active vLLM platform uses for device control - CUDA_VISIBLE_DEVICES on NVIDIA,
ZE_AFFINITY_MASK on Intel XPU - obtained from vllm.platforms.current_platform rather than
hard-coded per vendor.

These tests drive the branch logic directly with a patched current_platform and ray, so they
run on any platform without a real accelerator. They prove:
  - NVIDIA behavior is unchanged (CUDA_VISIBLE_DEVICES set from the Ray-assigned id).
  - Intel XPU sets ZE_AFFINITY_MASK (and not CUDA_VISIBLE_DEVICES).
  - The ray-executor branch clears the active platform's control var plus the CUDA/ROCm/HIP set.
  - An empty ray.get_gpu_ids() raises an actionable error instead of an opaque IndexError.

They do NOT prove real multi-accelerator pinning (two engines landing on two different
devices) - that requires a real multi-GPU/multi-XPU host.
"""

from unittest.mock import patch

import pytest

import openrlhf.trainer.ray.vllm_engine as vllm_engine

# RolloutRayActor is @ray.remote-decorated (an ActorClass, not a plain type). Reach the
# underlying Python class so we can call the plain method without a live Ray actor.
_meta = getattr(vllm_engine.RolloutRayActor, "__ray_metadata__", None)
RolloutRayActor = _meta.modified_class if _meta is not None else vllm_engine.RolloutRayActor


def _call(backend, platform_env_var, gpu_ids, noset=True, bundle_indices=None, num_gpus=1):
    """Invoke _configure_device_env with a patched device-var resolver + ray, on a clean env."""
    clean_env = {}  # isolated environment for the assertion
    with patch.dict("os.environ", clean_env, clear=True), patch(
        "openrlhf.trainer.ray.vllm_engine._resolve_device_control_env_var", return_value=platform_env_var
    ), patch("openrlhf.trainer.ray.vllm_engine.ray") as fake_ray, patch(
        "openrlhf.trainer.ray.vllm_engine.ray_noset_visible_devices", return_value=noset
    ):
        fake_ray.get_gpu_ids.return_value = gpu_ids
        RolloutRayActor._configure_device_env(object.__new__(RolloutRayActor), backend, bundle_indices, num_gpus)
        import os

        return dict(os.environ)


# --- non-ray branch (RAY_NOSET set): pin via the platform's control var ------------------
def test_nvidia_sets_cuda_visible_devices():
    env = _call(backend="mp", platform_env_var="CUDA_VISIBLE_DEVICES", gpu_ids=[3])
    assert env.get("CUDA_VISIBLE_DEVICES") == "3"
    assert "ZE_AFFINITY_MASK" not in env  # no XPU var leaks onto the CUDA path


def test_xpu_sets_ze_affinity_mask():
    env = _call(backend="mp", platform_env_var="ZE_AFFINITY_MASK", gpu_ids=[2])
    assert env.get("ZE_AFFINITY_MASK") == "2"
    assert "CUDA_VISIBLE_DEVICES" not in env  # only the active platform's var is set


def test_empty_gpu_ids_raises_actionable_error():
    with pytest.raises(RuntimeError, match="did not assign an accelerator"):
        _call(backend="mp", platform_env_var="ZE_AFFINITY_MASK", gpu_ids=[])


def test_unresolvable_control_var_raises():
    # Non-GPU / undetectable platform -> resolver returns None -> clear error, both branches.
    with pytest.raises(RuntimeError, match="Could not determine the device-control"):
        _call(backend="mp", platform_env_var=None, gpu_ids=[0])


def test_xpu_default_ray_sets_mask_when_var_absent():
    # Default Ray (noset=False): Ray sets CUDA_VISIBLE_DEVICES but not ZE_AFFINITY_MASK.
    # Since ZE_AFFINITY_MASK is absent, we must still set it (the XPU-pinning gap).
    env = _call(backend="mp", platform_env_var="ZE_AFFINITY_MASK", gpu_ids=[2], noset=False)
    assert env.get("ZE_AFFINITY_MASK") == "2"


def test_nvidia_default_ray_leaves_existing_var_untouched():
    # Default Ray (noset=False) on NVIDIA: Ray already set CUDA_VISIBLE_DEVICES, so the var is
    # present -> we must NOT overwrite it (Ray owns it in this mode).
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7"}, clear=True), patch(
        "openrlhf.trainer.ray.vllm_engine._resolve_device_control_env_var", return_value="CUDA_VISIBLE_DEVICES"
    ), patch("openrlhf.trainer.ray.vllm_engine.ray") as fake_ray, patch(
        "openrlhf.trainer.ray.vllm_engine.ray_noset_visible_devices", return_value=False
    ):
        import os

        fake_ray.get_gpu_ids.return_value = [3]
        RolloutRayActor._configure_device_env(object.__new__(RolloutRayActor), "mp", None, 1)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"  # unchanged, not overwritten with 3


# --- ray-executor branch: clear inherited visibility so the executor can assign ----------
def test_ray_backend_clears_active_platform_var():
    # Pre-existing restrictions must be popped, including the active platform's control var.
    with patch.dict(
        "os.environ",
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "ZE_AFFINITY_MASK": "1",
            "ROCR_VISIBLE_DEVICES": "1",
            "HIP_VISIBLE_DEVICES": "1",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:1",
        },
        clear=True,
    ), patch(
        "openrlhf.trainer.ray.vllm_engine._resolve_device_control_env_var", return_value="ZE_AFFINITY_MASK"
    ), patch(
        "openrlhf.trainer.ray.vllm_engine.ray"
    ), patch(
        "openrlhf.trainer.ray.vllm_engine.ray_noset_visible_devices", return_value=True
    ):
        import os

        RolloutRayActor._configure_device_env(object.__new__(RolloutRayActor), "ray", None, 1)
        for var in (
            "CUDA_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ONEAPI_DEVICE_SELECTOR",
        ):
            assert var not in os.environ, f"{var} should have been cleared in the ray branch"


# --- placement-group vars preserved regardless of platform -------------------------------
def test_bundle_indices_env_preserved():
    env = _call(backend="mp", platform_env_var="ZE_AFFINITY_MASK", gpu_ids=[0], bundle_indices=[4, 5], num_gpus=2)
    assert env.get("VLLM_RAY_PER_WORKER_GPUS") == "2"
    assert env.get("VLLM_RAY_BUNDLE_INDICES") == "4,5"
