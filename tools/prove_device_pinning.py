#!/usr/bin/env python
"""Demonstrate OpenRLHF's vLLM device-pinning logic.

Part A prints the real vLLM platform and its device-control env var on this machine.
Part B runs the same branch logic as RolloutRayActor._configure_device_env against a
simulated platform and Ray, for every scenario, and prints the resulting environment.

    python tools/prove_device_pinning.py          # Part A + Part B
    python tools/prove_device_pinning.py --real    # Part A only
"""

import argparse
import os
import sys


_WATCHED = ["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK",
            "VLLM_RAY_PER_WORKER_GPUS", "VLLM_RAY_BUNDLE_INDICES"]


# Branch logic copied verbatim from RolloutRayActor._configure_device_env, with its
# dependencies passed as arguments so any scenario can be exercised.
def configure_device_env(*, backend, device_control_env_var, ray_gpu_ids, noset, bundle_indices, num_gpus):
    if backend == "ray":
        for env_var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", device_control_env_var):
            os.environ.pop(env_var, None)
    elif noset:
        gpu_ids = ray_gpu_ids
        if not gpu_ids:
            raise RuntimeError("Ray did not assign an accelerator to the vLLM actor")
        if not device_control_env_var:
            raise RuntimeError("The active vLLM platform does not define a device-control environment variable")
        os.environ[device_control_env_var] = str(gpu_ids[0])

    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))


def _snapshot():
    return {k: os.environ.get(k) for k in _WATCHED}


def _print_env(label, env):
    print(f"    {label}")
    items = {k: v for k, v in env.items() if v is not None}
    if items:
        for k, v in items.items():
            print(f"        {k} = {v}")
    else:
        print("        (none of the watched vars set)")


def part_a_real():
    print("=" * 78)
    print("PART A - real vLLM platform on this machine")
    print("=" * 78)
    try:
        from vllm.platforms import current_platform
    except Exception as e:  # noqa: BLE001
        print(f"  Could not import vllm.platforms.current_platform: {e}")
        return

    cls = type(current_platform)
    print(f"  platform               : {cls.__module__}.{cls.__name__}")
    print(f"  device_control_env_var : {current_platform.device_control_env_var!r}")
    print(f"  ray_device_key         : {getattr(current_platform, 'ray_device_key', 'n/a')!r}")
    print()


def _run_scenario(title, *, backend, device_control_env_var, ray_gpu_ids, noset,
                  bundle_indices=None, num_gpus=1, preset_env=None):
    print("-" * 78)
    print(title)
    print("-" * 78)
    print(f"    inputs: backend={backend!r}  platform_var={device_control_env_var!r}  "
          f"ray_gpu_ids={ray_gpu_ids}  noset={noset}  bundle_indices={bundle_indices}  num_gpus={num_gpus}")

    saved = dict(os.environ)
    try:
        os.environ.clear()
        if preset_env:
            os.environ.update(preset_env)
            _print_env("BEFORE:", _snapshot())
        try:
            configure_device_env(
                backend=backend, device_control_env_var=device_control_env_var,
                ray_gpu_ids=ray_gpu_ids, noset=noset, bundle_indices=bundle_indices, num_gpus=num_gpus,
            )
            _print_env("AFTER :", _snapshot())
        except RuntimeError as e:
            print(f"    RAISED (as intended): RuntimeError: {e}")
    finally:
        os.environ.clear()
        os.environ.update(saved)
    print()


def part_b_simulated():
    print("=" * 78)
    print("PART B - simulated: what _configure_device_env sets per scenario")
    print("=" * 78)
    print()

    CUDA = "CUDA_VISIBLE_DEVICES"
    XPU = "ZE_AFFINITY_MASK"

    _run_scenario("1. NVIDIA, single GPU, non-ray (assigned GPU 0)",
                  backend="mp", device_control_env_var=CUDA, ray_gpu_ids=[0], noset=True)
    _run_scenario("2. NVIDIA, multi-GPU, non-ray (assigned GPU 3 of 0-7)",
                  backend="mp", device_control_env_var=CUDA, ray_gpu_ids=[3], noset=True)
    _run_scenario("3. Intel XPU, single GPU, non-ray (assigned XPU 0)",
                  backend="mp", device_control_env_var=XPU, ray_gpu_ids=[0], noset=True)
    _run_scenario("4. Intel XPU, multi-GPU, non-ray (assigned XPU 2 of 0-3)",
                  backend="mp", device_control_env_var=XPU, ray_gpu_ids=[2], noset=True)
    _run_scenario("5. NVIDIA, ray backend (pin cleared)",
                  backend="ray", device_control_env_var=CUDA, ray_gpu_ids=[3], noset=True,
                  preset_env={CUDA: "3", "ROCR_VISIBLE_DEVICES": "3"})
    _run_scenario("6. Intel XPU, ray backend (pin cleared, incl. ZE_AFFINITY_MASK)",
                  backend="ray", device_control_env_var=XPU, ray_gpu_ids=[2], noset=True,
                  preset_env={CUDA: "2", XPU: "2"})
    _run_scenario("7. Intel XPU, non-ray, tensor-parallel bundle",
                  backend="mp", device_control_env_var=XPU, ray_gpu_ids=[1], noset=True,
                  bundle_indices=[4, 5], num_gpus=2)
    _run_scenario("8. Error: Ray assigned no accelerator (empty gpu_ids)",
                  backend="mp", device_control_env_var=XPU, ray_gpu_ids=[], noset=True)
    _run_scenario("9. Error: platform declares no device-control var",
                  backend="mp", device_control_env_var="", ray_gpu_ids=[0], noset=True)
    _run_scenario("10. noset=False, non-ray (Ray manages visibility; we set nothing)",
                  backend="mp", device_control_env_var=XPU, ray_gpu_ids=[2], noset=False,
                  preset_env={XPU: "already-set-by-ray"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="only Part A")
    args = ap.parse_args()

    part_a_real()
    if not args.real:
        part_b_simulated()
    return 0


if __name__ == "__main__":
    sys.exit(main())
