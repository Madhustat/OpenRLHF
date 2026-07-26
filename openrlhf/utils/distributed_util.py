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

    When ``backend`` is None (unset), auto-select NCCL/RCCL for CUDA/ROCm builds and gloo for
    other accelerator builds (e.g. Intel XPU). An explicitly configured backend is honored and
    validated strictly: it is never silently replaced, and an impossible combination (e.g.
    'nccl' on a build without CUDA/ROCm) raises a clear error.
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
            "or None for automatic selection"
        )

    if backend == "nccl" and not is_nccl_platform:
        raise RuntimeError(
            "vLLM sync_backend='nccl' requires a CUDA or ROCm build. "
            "Use sync_backend='gloo', or omit the option to auto-detect the backend."
        )

    return backend


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend=None):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.

    ``backend`` is the vLLM weight-sync backend. None auto-detects per platform; an explicit
    value is validated strictly (see resolve_vllm_sync_backend).
    """
    backend = resolve_vllm_sync_backend(backend)

    if backend == "nccl":
        # NVIDIA/AMD path: vLLM's real GPU-to-GPU communicator over NCCL/RCCL.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        logger.info("vLLM weight-sync backend: pynccl")
        return PyNcclCommunicator(pg, device=device)

    if backend == "gloo":
        # No NCCL/RCCL on this platform (e.g. Intel XPU). PyNcclCommunicator silently sets
        # disabled=True in that case and every broadcast() becomes a no-op with no error -
        # confirmed via live instrumentation (see CORRECTION_weight_sync_not_real.md).
        #
        # StatelessProcessGroup.broadcast() (TCP-store based) is correct but pickles the WHOLE
        # tensor into an in-memory TCPStore per parameter; under --train.colocate_all this
        # measured ~2.3s/param (809s for one 0.5B weight-sync round). A real gloo ProcessGroup
        # is a genuine collective backend (~5-6ms/call) and avoids that stall. gloo doesn't
        # support XPU tensors directly, so _GlooBroadcastCommunicator stages through CPU.
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            master_address, master_port, rank, world_size, backend="gloo"
        )
        logger.info("vLLM weight-sync backend: gloo (CPU staging)")
        return _GlooBroadcastCommunicator(pg, device)

    # Defensive: resolve_vllm_sync_backend already rejects anything else.
    raise AssertionError(f"Unexpected resolved backend: {backend!r}")
