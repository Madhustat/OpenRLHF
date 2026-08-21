import logging

logger = logging.getLogger(__name__)


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and accelerator operations."""
    import torch

    torch.distributed.barrier()
    torch.accelerator.synchronize()


class _GlooBroadcastCommunicator:
    """Broadcast accelerator tensors through a CPU gloo process group (control/baseline path).

    Staged to CPU for the gloo collective, then copied back into the original accelerator
    tensor (storage preserved). Returns the same tensor, matching the PyNcclCommunicator
    interface. gloo does not support XPU tensors directly, hence the CPU staging.
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

    The tensor stays on the XPU and XCCL (Intel oneCCL) moves it XPU->XPU. Broadcasts in
    place and returns the same tensor, matching the PyNcclCommunicator interface.
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
    """Build a stateless XCCL ProcessGroup for XPU, mirroring vLLM's NCCL stateless pattern.

    vLLM's XPU platform does not implement stateless_init_device_torch_dist_pg (raises
    NotImplementedError), so we construct the ProcessGroup + XCCL backend directly here.

    ``device`` is this rank's own XPU (e.g. torch.device("xpu", idx)). We bind it explicitly
    before building the group and register the backend for that specific device index -
    without this, ProcessGroupXCCL warns "device ... is currently unknown" and can HANG
    because the rank->device mapping is undefined.
    """
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, ProcessGroupXCCL

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


def dist_xccl_available():
    import torch.distributed as dist

    return hasattr(dist, "is_xccl_available") and dist.is_xccl_available()


def _physical_xpu_count():
    """Number of physical Intel GPUs on the host, INDEPENDENT of device-visibility masks.

    Under Ray each actor is pinned to one device via ZE_AFFINITY_MASK, so
    torch.xpu.device_count() returns 1 inside the actor even on a multi-XPU box. The
    weight-sync group spans one rank per actor, each on its OWN physical XPU, so we count
    Intel DRM render nodes (/sys/class/drm/renderD*/device/vendor == 0x8086), which the
    affinity mask does not hide. Returns 0 on any error.
    """
    import glob
    import os

    count = 0
    try:
        for render in glob.glob("/sys/class/drm/renderD*"):
            try:
                with open(os.path.join(render, "device", "vendor")) as f:
                    if f.read().strip().lower() == "0x8086":
                        count += 1
            except OSError:
                continue
    except Exception:
        return 0
    return count


def resolve_vllm_sync_backend(backend=None):
    """Resolve and validate the vLLM weight-sync backend.

    None auto-selects per platform: NCCL/RCCL on CUDA/ROCm, XCCL on Intel XPU. gloo is a
    CPU-staged fallback valid on any accelerator (kept as the control/default XPU path).
    An explicit value is validated strictly - an impossible combo raises.
    """
    import torch

    is_nccl_platform = bool(getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None))
    is_xpu_platform = bool(getattr(torch.version, "xpu", None)) and dist_xccl_available()
    xpu_count = max(_physical_xpu_count(), torch.xpu.device_count()) if is_xpu_platform else 0

    if backend is None:
        effective_backend = "nccl" if is_nccl_platform else "gloo"
        logger.info("vLLM weight-sync backend auto-selected: %s (xpu_count=%d)", effective_backend, xpu_count)
        return effective_backend

    if backend not in {"nccl", "gloo", "xccl"}:
        raise ValueError(f"Unsupported vLLM weight-sync backend {backend!r}; expected 'nccl', 'gloo', 'xccl', or None")

    if backend == "nccl" and not is_nccl_platform:
        raise RuntimeError("vLLM sync_backend='nccl' requires a CUDA or ROCm build.")

    if backend == "xccl" and not is_xpu_platform:
        raise RuntimeError("vLLM sync_backend='xccl' requires an Intel XPU build with XCCL support.")

    if backend == "xccl" and xpu_count < 2:
        # oneCCL needs one rank per physical device; XCCL is only correct with >=2 physical XPUs.
        raise RuntimeError(
            f"sync_backend='xccl' needs >=2 physical XPUs (one rank per device); found {xpu_count}."
        )

    return backend


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend=None):
    """Create the DeepSpeed<->vLLM weight-sync communicator for the resolved backend.

    ``backend`` None auto-detects per platform; an explicit value is validated strictly
    (see resolve_vllm_sync_backend).
    """
    backend = resolve_vllm_sync_backend(backend)

    if backend == "nccl":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        logger.info("vLLM weight-sync backend: pynccl")
        return PyNcclCommunicator(pg, device=device)

    if backend == "xccl":
        # Intel XPU: broadcast weights directly XPU->XPU over a stateless XCCL process group.
        pg = _init_xccl_process_group(master_address, master_port, rank, world_size, device)
        logger.info("vLLM weight-sync backend: xccl (direct XPU-to-XPU)")
        return _XcclBroadcastCommunicator(pg, device)

    if backend == "gloo":
        # CPU-staged control/baseline path (any accelerator). A real gloo ProcessGroup is a
        # genuine collective backend; _GlooBroadcastCommunicator stages XPU tensors through CPU.
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            master_address, master_port, rank, world_size, backend="gloo"
        )
        logger.info("vLLM weight-sync backend: gloo (CPU staging)")
        return _GlooBroadcastCommunicator(pg, device)

    raise AssertionError(f"Unexpected resolved backend: {backend!r}")
