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


class _XcclBroadcastCommunicator:
    """Broadcast XPU tensors directly device-to-device over a stateless XCCL process group.

    Unlike the gloo fallback, no CPU staging: the tensor stays on the XPU and XCCL (Intel
    oneCCL) moves it XPU->XPU. Broadcasts in place and returns the same tensor, matching the
    PyNcclCommunicator / _GlooBroadcastCommunicator interface.
    """

    def __init__(self, pg, device):
        self.pg = pg
        self.device = device

    def broadcast(self, tensor, src, stream=None):
        # stream is accepted for interface parity; XCCL uses the current XPU stream.
        from torch.distributed.distributed_c10d import BroadcastOptions

        opts = BroadcastOptions()
        opts.rootRank = src
        work = self.pg.broadcast([tensor], opts)
        work.wait()
        return tensor


def _init_xccl_process_group(master_address, master_port, rank, world_size, device):
    """Build a stateless XCCL ProcessGroup for XPU, mirroring vLLM's gloo/NCCL stateless
    pattern. Needed because vLLM's XPU platform does not implement
    stateless_init_device_torch_dist_pg (it raises NotImplementedError), so we construct the
    ProcessGroup + XCCL backend directly here instead of going through vLLM's helper.

    ``device`` is this rank's own XPU (e.g. torch.device("xpu", idx)). We bind it explicitly
    before/while building the group and register the backend for that specific device index -
    without this, ProcessGroupXCCL warns "device ... is currently unknown" and can HANG because
    the rank->device mapping is undefined. The backend is registered for the exact xpu:index
    rather than a generic torch.device("xpu").
    """
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, ProcessGroupXCCL

    # Normalize to an indexed xpu device and make it current before constructing the backend,
    # so XCCL/Level-Zero binds this rank to a definite device.
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "xpu":
        raise RuntimeError(f"xccl weight-sync requires an XPU device, got {device!r}")
    device_index = device.index if device.index is not None else torch.xpu.current_device()
    device = torch.device("xpu", device_index)
    torch.xpu.set_device(device_index)

    timeout = timedelta(seconds=1800)
    store = dist.TCPStore(master_address, master_port, world_size, is_master=(rank == 0), timeout=timeout)
    prefix_store = PrefixStore("openrlhf-xccl", store)

    pg = ProcessGroup(prefix_store, rank, world_size)
    # Bind the group to this rank's device so collectives have a defined rank->device mapping.
    try:
        pg.bound_device_id = device
    except Exception:
        # Older torch builds may not expose bound_device_id; set_device above is the fallback.
        pass
    backend_class = ProcessGroupXCCL(prefix_store, rank, world_size)
    backend_type = ProcessGroup.BackendType.XCCL
    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()
    pg._register_backend(device, backend_type, backend_class)
    return pg


def resolve_vllm_sync_backend(backend=None):
    """Resolve and validate the vLLM weight-sync backend.

    When ``backend`` is None (unset), auto-select per platform: NCCL/RCCL for CUDA/ROCm, XCCL
    for Intel XPU (direct device-to-device), and gloo as a last-resort fallback. An explicitly
    configured backend is honored and validated strictly: it is never silently replaced, and an
    impossible combination (e.g. 'nccl' on a build without CUDA/ROCm) raises a clear error.
    """
    import torch

    is_nccl_platform = bool(getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None))
    is_xpu_platform = bool(getattr(torch.version, "xpu", None)) and dist_xccl_available()

    if backend is None:
        # Auto-detect a SAFE default: nccl on CUDA/ROCm, else gloo. gloo (CPU-staged) works on
        # any XPU config including a single device. XCCL (direct XPU->XPU) is faster but is
        # opt-in via --vllm.sync_backend xccl, because it requires distinct physical devices
        # per rank and hangs when the trainer and vLLM engine share one XPU (colocated,
        # single-GPU). Auto-selecting xccl would regress single-XPU users, so we don't.
        effective_backend = "nccl" if is_nccl_platform else "gloo"
        logger.info("vLLM weight-sync backend auto-selected: %s", effective_backend)
        return effective_backend

    if backend not in {"nccl", "gloo", "xccl"}:
        raise ValueError(
            f"Unsupported vLLM weight-sync backend {backend!r}; expected 'nccl', 'gloo', "
            "'xccl', or None for automatic selection"
        )

    if backend == "nccl" and not is_nccl_platform:
        raise RuntimeError(
            "vLLM sync_backend='nccl' requires a CUDA or ROCm build. "
            "Use sync_backend='xccl' (Intel XPU, multi-device) or 'gloo', or omit to auto-detect."
        )

    if backend == "xccl" and not is_xpu_platform:
        raise RuntimeError(
            "vLLM sync_backend='xccl' requires an Intel XPU build with XCCL support. "
            "Use sync_backend='nccl' (CUDA/ROCm) or 'gloo', or omit the option to auto-detect."
        )

    if backend == "xccl" and torch.xpu.device_count() < 2:
        # XCCL is a device-to-device collective. With the trainer and vLLM engine sharing a
        # single physical XPU it deadlocks, so fall back to gloo (CPU-staged, single-device
        # safe) rather than hang. XCCL is only engaged when >=2 physical XPUs are present.
        logger.warning(
            "sync_backend='xccl' requested but only %d XPU visible; XCCL needs >=2 physical "
            "devices (single-device XCCL hangs). Falling back to gloo.",
            torch.xpu.device_count(),
        )
        return "gloo"

    return backend


def dist_xccl_available():
    import torch.distributed as dist

    return hasattr(dist, "is_xccl_available") and dist.is_xccl_available()


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

    if backend == "xccl":
        # Intel XPU: broadcast weights directly XPU->XPU over a stateless XCCL process group,
        # no CPU staging. vLLM's XPU platform doesn't implement stateless_init_device_torch_dist_pg,
        # so we build the XCCL ProcessGroup ourselves (see _init_xccl_process_group). device is
        # this rank's own XPU - passed through so XCCL binds a definite rank->device mapping.
        pg = _init_xccl_process_group(master_address, master_port, rank, world_size, device)
        logger.info("vLLM weight-sync backend: xccl (direct XPU-to-XPU)")
        return _XcclBroadcastCommunicator(pg, device)

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
