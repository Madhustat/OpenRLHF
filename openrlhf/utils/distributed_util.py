from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and accelerator operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All accelerator operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.accelerator.synchronize()


def resolve_vllm_sync_backend(backend, colocated):
    import torch
    from packaging import version

    nccl_ok = bool(torch.version.cuda or torch.version.hip)
    xccl_ok = (
        bool(getattr(torch.version, "xpu", None))
        and torch.distributed.is_xccl_available()
        and version.parse(torch.__version__).release >= (2, 13)
    )
    if backend == "auto":
        backend = "nccl" if nccl_ok else "xccl" if xccl_ok and not colocated else "gloo"
    elif backend == "nccl" and not nccl_ok:
        raise ValueError("--vllm.sync_backend nccl needs CUDA or ROCm; on Intel XPU use auto, xccl or gloo")
    elif backend == "xccl" and not xccl_ok:
        raise ValueError("--vllm.sync_backend xccl needs an Intel XPU build of torch >= 2.13 with XCCL; use gloo")
    elif backend == "xccl" and colocated:
        logger.warning("XCCL weight sync between processes sharing an XPU (colocate_all) is experimental")
    logger.info(f"vLLM weight-sync backend: {backend}")
    return backend


class _GlooBroadcastCommunicator:
    def __init__(self, pg):
        self.pg = pg

    def broadcast(self, tensor, src, stream=None):
        import torch

        cpu_tensor = tensor.cpu() if self.pg.rank() == src else torch.empty(tensor.shape, dtype=tensor.dtype)
        self.pg.broadcast(cpu_tensor, src).wait()
        if self.pg.rank() != src:
            tensor.copy_(cpu_tensor)
        return tensor


class _XcclBroadcastCommunicator:
    def __init__(self, pg):
        self.pg = pg

    def broadcast(self, tensor, src, stream=None):
        from torch.distributed.distributed_c10d import BroadcastOptions

        opts = BroadcastOptions()
        opts.rootRank = src
        self.pg.broadcast([tensor], opts).wait()
        return tensor


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend="nccl"):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    if backend == "nccl":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        return PyNcclCommunicator(pg, device=device)
    if backend == "gloo":
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            master_address, master_port, rank, world_size, backend="gloo"
        )
        return _GlooBroadcastCommunicator(pg)

    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, ProcessGroupXCCL

    torch.xpu.set_device(device)
    store = dist.TCPStore(
        master_address, master_port, world_size, is_master=rank == 0, timeout=timedelta(seconds=1800)
    )
    prefix_store = PrefixStore("openrlhf-xccl", store)
    pg = ProcessGroup(prefix_store, rank, world_size)
    pg.bound_device_id = device
    xccl = ProcessGroupXCCL(prefix_store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.XCCL)
    xccl._set_sequence_number_for_group()
    pg._register_backend(device, ProcessGroup.BackendType.XCCL, xccl)
    return _XcclBroadcastCommunicator(pg)
