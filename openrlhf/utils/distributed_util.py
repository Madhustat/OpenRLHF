import logging

logger = logging.getLogger(__name__)


def torch_dist_barrier_and_accelerator_sync():
    """Synchronize distributed ranks and the current accelerator.
    1. All distributed processes reach this point (barrier)
    2. All queued accelerator operations complete (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.accelerator.synchronize()


class _GlooBroadcastCommunicator:
    """Broadcast accelerator tensors through a CPU gloo process group.

    Tensors are staged to CPU for the gloo collective and copied back into the original
    accelerator tensor. The original tensor storage is preserved (copy_), and the same
    tensor is returned for interface compatibility with PyNcclCommunicator. gloo does not
    support XPU tensors directly, hence the CPU staging.
    """

    def __init__(self, pg, device):
        self.pg = pg
        self.device = device

    def broadcast(self, tensor, src, stream=None):
        # stream is accepted for interface parity with PyNcclCommunicator; gloo ignores it.
        import torch

        cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        work = self.pg.broadcast(cpu_tensor, src)
        work.wait()
        with torch.no_grad():
            tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        return tensor


def resolve_vllm_sync_backend(backend=None):
    """Resolve and validate the vLLM weight-sync backend.

    When ``backend`` is None (unset), auto-select the appropriate backend for the platform:
      - CUDA / ROCm -> nccl  (native GPU-to-GPU via PyNcclCommunicator)
      - Intel XPU   -> gloo  (CPU-staged; gloo works on all platforms)
      - other       -> gloo

    An explicit value is validated strictly:
      - 'nccl' on a non-CUDA/ROCm build raises RuntimeError
      - unknown values raise ValueError

    The effective backend is logged once so the selected path is visible in logs.
    """
    import torch

    is_nccl_platform = bool(getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None))

    if backend is None:
        effective_backend = "nccl" if is_nccl_platform else "gloo"
        logger.info("vLLM weight-sync backend auto-selected: %s", effective_backend)
        return effective_backend

    if backend not in {"nccl", "gloo"}:
        raise ValueError(
            f"Unsupported vLLM weight-sync backend {backend!r}; expected 'nccl', 'gloo', "
            "or None for automatic selection."
        )

    if backend == "nccl" and not is_nccl_platform:
        raise RuntimeError(
            "vLLM sync_backend='nccl' requires a CUDA or ROCm build. "
            "Use sync_backend='gloo' for Intel XPU, or omit to auto-detect."
        )

    return backend


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend=None):
    """Create a stateless process group for actor-to-vLLM weight synchronisation.

    Dispatches to the appropriate communicator based on the resolved backend:
      - nccl  -> PyNcclCommunicator  (CUDA/ROCm; real GPU-to-GPU, no CPU staging)
      - gloo  -> _GlooBroadcastCommunicator  (CPU-staged; works on any platform including XPU)

    ``backend`` is validated and resolved by resolve_vllm_sync_backend before dispatch.
    Pass None to auto-select the platform-appropriate default.
    """
    backend = resolve_vllm_sync_backend(backend)

    if backend == "nccl":
        # CUDA / ROCm: use vLLM's real GPU-to-GPU communicator over NCCL/RCCL.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        logger.info("vLLM weight-sync backend: nccl")
        return PyNcclCommunicator(pg, device=device)

    # gloo: CPU-staged broadcast. PyNcclCommunicator silently disables itself on platforms
    # without NCCL (e.g. Intel XPU) — every broadcast() becomes a no-op and vLLM keeps
    # generating from the initial weights throughout training. _GlooBroadcastCommunicator
    # stages tensors to CPU, runs the gloo collective, and copies back in-place (copy_()).
    from vllm.distributed.utils import stateless_init_torch_distributed_process_group

    pg = stateless_init_torch_distributed_process_group(
        master_address, master_port, rank, world_size, backend="gloo"
    )
    logger.info("vLLM weight-sync backend: gloo (CPU staging)")
    return _GlooBroadcastCommunicator(pg, device)
