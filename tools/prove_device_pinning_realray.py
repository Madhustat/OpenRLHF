#!/usr/bin/env python
"""Real-Ray proof of the vLLM device-pinning branches.

Starts (or connects to) Ray, launches one GPU actor per available GPU concurrently, and
inside each actor calls the real ray.get_gpu_ids() and runs the real
RolloutRayActor._configure_device_env, printing the environment before/after.

On a multi-GPU box this shows each actor pinned to a different device.

    ray start --head --num-gpus=N      # optional; script self-starts if Ray is not running
    python tools/prove_device_pinning_realray.py
"""
import os
import sys

import ray

import openrlhf.trainer.ray.vllm_engine as vllm_engine
from openrlhf.trainer.ray.utils import ray_noset_visible_devices

_meta = getattr(vllm_engine.RolloutRayActor, "__ray_metadata__", None)
_RealActorClass = _meta.modified_class if _meta is not None else vllm_engine.RolloutRayActor

_WATCHED = ["CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK", "ONEAPI_DEVICE_SELECTOR"]


def _env_snapshot():
    return {k: os.environ.get(k) for k in _WATCHED}


@ray.remote(num_gpus=1)
class ProbeActor:
    def probe(self, backend):
        from vllm.platforms import current_platform

        report = {
            "gpu_ids": ray.get_gpu_ids(),
            "noset": ray_noset_visible_devices(),
            "platform": type(current_platform).__name__,
            "device_control_env_var": current_platform.device_control_env_var,
            "env_before": _env_snapshot(),
        }
        try:
            _RealActorClass._configure_device_env(object.__new__(_RealActorClass), backend, None, 1)
            report["result"] = "ok"
        except Exception as e:  # noqa: BLE001
            report["result"] = f"{type(e).__name__}: {e}"
        report["env_after"] = _env_snapshot()
        return report


def _print_report(title, r):
    def show(label, env):
        items = {k: v for k, v in env.items() if v is not None}
        print(f"    {label}: {items if items else '(none set)'}")

    print("-" * 78)
    print(title)
    print("-" * 78)
    print(f"    ray.get_gpu_ids()            = {r['gpu_ids']}")
    print(f"    ray_noset_visible_devices()  = {r['noset']}")
    print(f"    platform                     = {r['platform']}")
    print(f"    device_control_env_var       = {r['device_control_env_var']!r}")
    show("env BEFORE", r["env_before"])
    print(f"    _configure_device_env(backend={r['_backend']!r}) -> {r['result']}")
    show("env AFTER ", r["env_after"])
    print()


def main():
    print("=" * 78)
    print("REAL-RAY PROOF: run _configure_device_env inside real GPU actors")
    print("=" * 78)

    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        # Self-start. Ray does not always auto-detect XPUs as GPUs, so honor an explicit
        # count via PROVE_NUM_GPUS (default 1). On a real cluster, prefer `ray start --head
        # --num-gpus=N` so Ray assigns actors across the real devices.
        n = int(os.environ.get("PROVE_NUM_GPUS", "1"))
        print(f"  No running Ray found - starting a local Ray with num_gpus={n}.")
        ray.init(num_gpus=n, ignore_reinit_error=True)

    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"  Ray GPUs available: {num_gpus}\n")
    if num_gpus == 0:
        print("  No GPUs in the Ray cluster; nothing to prove.")
        return 1

    for backend in ("mp", "ray"):
        print(f"### backend={backend!r}: launching {num_gpus} actor(s) concurrently\n")
        # Launch all actors first so Ray must assign them to different GPUs at once.
        actors = [ProbeActor.remote() for _ in range(num_gpus)]
        reports = ray.get([a.probe.remote(backend) for a in actors])
        for i, r in enumerate(reports):
            r["_backend"] = backend
            _print_report(f"actor {i}  backend={backend!r}", r)
        for a in actors:
            ray.kill(a)

        if backend == "mp":
            assigned = sorted(r["gpu_ids"][0] for r in reports if r["gpu_ids"])
            print(f"  GPUs assigned across actors: {assigned}"
                  f"  ->  {'DISTINCT (no collision)' if len(set(assigned)) == len(assigned) else 'COLLISION'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
