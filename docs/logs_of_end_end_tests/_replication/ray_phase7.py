import os
import ray
import torch

NUM_GPUS = torch.xpu.device_count()
print("Driver: detected XPU device count =", NUM_GPUS)

os.environ.setdefault("CCL_LOG_LEVEL", "info")
os.environ.setdefault("NCCL_DEBUG", "WARN")

ray.init(num_gpus=NUM_GPUS, ignore_reinit_error=True,
         runtime_env={"env_vars": {"CCL_LOG_LEVEL": "info", "NCCL_DEBUG": "WARN"}})
print("Ray initialized. Cluster resources:", ray.cluster_resources())


@ray.remote(num_gpus=1)
class XPUWorker:
    def check(self):
        import torch
        info = {
            "xpu_available": torch.xpu.is_available(),
            "xpu_count_in_worker": torch.xpu.device_count(),
            "device_name": torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
            "CCL_LOG_LEVEL": os.environ.get("CCL_LOG_LEVEL"),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG"),
        }
        # do a real tensor op on the XPU inside the worker
        x = torch.randn(1024, 1024, device="xpu")
        y = (x @ x).sum().item()
        info["matmul_ok"] = bool(y == y)  # not NaN
        return info


worker = XPUWorker.remote()
result = ray.get(worker.check.remote())
print("Worker result:", result)

assert result["xpu_available"], "XPU not visible inside Ray worker"
assert result["matmul_ok"], "XPU matmul failed inside Ray worker"
assert result["CCL_LOG_LEVEL"] == "info", "CCL_LOG_LEVEL not propagated"
assert result["NCCL_DEBUG"] == "WARN", "NCCL_DEBUG not propagated"
print("PHASE7_RESULT=PASS")
ray.shutdown()
