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
    """Broadcast XPU tensors device-to-device over a stateless XCCL process group (no CPU staging).

    Broadcasts in place and returns the same tensor, like PyNcclCommunicator and
    _GlooBroadcastCommunicator.
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
    """Stateless XCCL ProcessGroup between the trainer and the vLLM workers.

    vLLM's XPU platform does not implement stateless_init_device_torch_dist_pg, so the group is
    built directly. It is bound to this rank's XPU (``device``) and the backend is registered for
    that exact device: without a defined rank -> device mapping ProcessGroupXCCL can hang.
    """
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import PrefixStore, ProcessGroup, ProcessGroupXCCL

    device = torch.device(device)
    if device.type != "xpu":
        raise RuntimeError(f"XCCL weight sync requires an XPU device, got {device!r}")
    device = torch.device("xpu", device.index if device.index is not None else torch.xpu.current_device())
    torch.xpu.set_device(device)

    store = dist.TCPStore(
        master_address, master_port, world_size, is_master=(rank == 0), timeout=timedelta(seconds=1800)
    )
    prefix_store = PrefixStore("openrlhf-xccl", store)
    pg = ProcessGroup(prefix_store, rank, world_size)
    pg.bound_device_id = device
    backend = ProcessGroupXCCL(prefix_store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.XCCL)
    backend._set_sequence_number_for_group()
    pg._register_backend(device, ProcessGroup.BackendType.XCCL, backend)
    return pg


class XcclUnavailableError(RuntimeError):
    """XCCL weight sync cannot be used (failed eligibility, placement or preflight check)."""


# oneCCL settings that break XCCL weight sync on Intel GPUs, each reproduced with a plain
# torch.distributed broadcast: GPU page faults on large tensors, a hang in the first broadcast,
# a segfault.
_UNSAFE_ONECCL_SETTINGS = {"CCL_ZE_CACHE_OPEN_IPC_HANDLES": "0", "FI_PROVIDER": "shm", "CCL_TOPO_ALGO": "0"}


def xccl_unavailable_reason(colocated=False):
    """Why auto mode does not select XCCL weight sync here, or None if it does.

    The six hard prerequisites of _xccl_environment_problem(), then the two auto-selection
    topology policies (colocated, single XPU). Static checks only - nothing here initializes an
    accelerator - so the backend can be chosen before any Ray actor exists. ``colocated``: vLLM
    shares XPUs with the trainer (colocate_all).
    """
    problem = _xccl_environment_problem()
    if problem:
        return problem
    if colocated:
        return "vLLM shares XPUs with the trainer (colocate_all); same-XPU XCCL weight sync is not validated yet"
    if _physical_xpu_count() < 2:
        return (
            "Auto selects Gloo for single-XPU and colocated execution because it is the currently validated "
            "default. Explicit XCCL remains available as an experimental option."
        )
    return None


def _xccl_environment_problem():
    """Prerequisites of XCCL weight sync that do not depend on the GPU layout."""
    import os
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    import torch
    from packaging import version

    if not getattr(torch.version, "xpu", None) or not dist_xccl_available():
        return "PyTorch is not an Intel XPU build with XCCL support"
    if version.parse(torch.__version__.split("+")[0]) < version.parse("2.13.0"):
        return (
            f"torch {torch.__version__} is older than 2.13 (XCCL broadcast crashes on 2.12, intel/torch-xpu-ops#4238)"
        )
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR") != "1":
        return (
            "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 is not set, so Ray restricts each actor to one "
            "XPU; oneCCL needs every communicator XPU visible in every process"
        )
    unsafe = {k: v for k, v in _UNSAFE_ONECCL_SETTINGS.items() if os.environ.get(k) == v}
    if unsafe:
        return f"oneCCL setting(s) known to break XCCL weight sync are set: {unsafe}"
    try:
        if version.parse(package_version("vllm")) < version.parse("0.31.0"):
            return "vLLM is older than 0.31 (its XPU worker ignores device_ids, needed for explicit XPU placement)"
    except PackageNotFoundError:
        return "vLLM is not installed"
    if os.environ.get("CCL_ZE_IPC_EXCHANGE", "pidfd") == "pidfd" and not _pidfd_getfd_permitted():
        return (
            "pidfd_getfd() is not permitted, and oneCCL needs it to share XPU memory between processes; "
            "in Docker, run with --cap-add SYS_PTRACE (or --security-opt seccomp=unconfined)"
        )
    return None


def _pidfd_getfd_permitted():
    """oneCCL shares XPU memory handles between processes with pidfd_getfd(2), which Docker's
    default seccomp profile denies without CAP_SYS_PTRACE (oneCCL then falls back to a path that
    can hang, uxlfoundation/oneCCL#217). Probes the call on a pipe of our own."""
    import ctypes
    import errno
    import os
    import platform

    if not hasattr(os, "pidfd_open") or platform.machine() != "x86_64":
        return True  # cannot probe here; the XCCL preflight broadcast still guards the real path
    sys_pidfd_getfd = 438
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = os.pidfd_open(os.getpid())
    read_end, write_end = os.pipe()
    try:
        fd = libc.syscall(sys_pidfd_getfd, pidfd, read_end, 0)
        if fd >= 0:
            os.close(fd)
            return True
        return ctypes.get_errno() not in (errno.EPERM, errno.EACCES, errno.ENOSYS)
    finally:
        for fd in (pidfd, read_end, write_end):
            os.close(fd)


def resolve_vllm_sync_backend(backend=None, colocated=False):
    """Resolve the DeepSpeed -> vLLM weight-sync backend.

    'auto' (or None): nccl on CUDA/ROCm; on Intel XPU, xccl when xccl_unavailable_reason() finds
    nothing missing, otherwise gloo (the reason is logged); gloo on other platforms.
    An explicit backend is honored or rejected, never replaced: 'nccl' needs CUDA/ROCm, 'xccl'
    needs every XPU prerequisite of xccl_unavailable_reason().
    """
    import torch

    is_nccl_platform = bool(getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None))

    if backend in (None, "auto"):
        if is_nccl_platform:
            effective_backend = "nccl"
        elif getattr(torch.version, "xpu", None):
            reason = xccl_unavailable_reason(colocated)
            effective_backend = "gloo" if reason else "xccl"
            if reason:
                logger.warning("vLLM weight sync uses gloo (CPU-staged), not XCCL: %s", reason)
        else:
            effective_backend = "gloo"
        logger.info("vLLM weight-sync backend auto-selected: %s", effective_backend)
        return effective_backend

    if backend not in {"nccl", "gloo", "xccl"}:
        raise ValueError(
            f"Unsupported vLLM weight-sync backend {backend!r}; expected 'auto', 'nccl', 'gloo' or 'xccl'"
        )

    if backend == "nccl" and not is_nccl_platform:
        raise RuntimeError(
            "vLLM sync_backend='nccl' requires a CUDA or ROCm build. "
            "Use sync_backend='xccl' or 'gloo' on Intel XPU, or 'auto'."
        )

    if backend == "xccl":
        # An explicit request may try same-XPU (colocated) XCCL; every hard prerequisite is required,
        # and without colocate_all the trainer and vLLM need different XPUs.
        reason = _xccl_environment_problem()
        if not reason and not colocated and _physical_xpu_count() < 2:
            reason = "fewer than two Intel XPUs found; XCCL weight sync needs the trainer and vLLM on different XPUs"
        if reason:
            raise XcclUnavailableError(
                f"vLLM sync_backend='xccl' cannot be used: {reason}. "
                "Use --vllm.sync_backend gloo (CPU-staged) or auto."
            )
        if colocated:
            logger.warning("vLLM sync_backend='xccl' with colocate_all: same-XPU XCCL weight sync is experimental")

    return backend


def dist_xccl_available():
    import torch.distributed as dist

    return hasattr(dist, "is_xccl_available") and dist.is_xccl_available()


def _physical_xpu_count():
    """Intel GPU render nodes on the host (/sys/class/drm/renderD*, vendor 0x8086), independent of
    visibility masks and without initializing the XPU runtime. 0 if sysfs cannot be read."""
    import glob

    count = 0
    for render in glob.glob("/sys/class/drm/renderD*"):
        try:
            with open(f"{render}/device/vendor") as f:
                count += f.read().strip().lower() == "0x8086"
        except OSError:
            continue
    return count


def stateless_init_process_group(master_address, master_port, rank, world_size, device, backend=None):
    """Create a stateless process group for actor-to-vLLM weight synchronisation.

    Dispatches to the appropriate communicator based on the resolved backend:
      - nccl  -> PyNcclCommunicator  (CUDA/ROCm; real GPU-to-GPU, no CPU staging)
      - xccl  -> _XcclBroadcastCommunicator  (Intel XPU; direct XPU-to-XPU, no CPU staging)
      - gloo  -> _GlooBroadcastCommunicator  (CPU-staged; works on any platform including XPU)

    ``backend`` is the backend resolved by resolve_vllm_sync_backend in the launcher; None or
    'auto' resolves it here.
    """
    if backend in (None, "auto"):
        backend = resolve_vllm_sync_backend(backend)

    if backend == "nccl":
        # CUDA / ROCm: use vLLM's real GPU-to-GPU communicator over NCCL/RCCL.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        logger.info("vLLM weight-sync backend: nccl")
        return PyNcclCommunicator(pg, device=device)

    if backend == "xccl":
        pg = _init_xccl_process_group(master_address, master_port, rank, world_size, device)
        logger.info("vLLM weight-sync backend: xccl (direct XPU-to-XPU)")
        return _XcclBroadcastCommunicator(pg, device)

    if backend != "gloo":
        raise ValueError(f"Unsupported vLLM weight-sync backend {backend!r}")

    # gloo: CPU-staged broadcast. PyNcclCommunicator silently disables itself on platforms
    # without NCCL (e.g. Intel XPU) — every broadcast() becomes a no-op and vLLM keeps
    # generating from the initial weights throughout training. _GlooBroadcastCommunicator
    # stages tensors to CPU, runs the gloo collective, and copies back in-place (copy_()).
    from vllm.distributed.utils import stateless_init_torch_distributed_process_group

    pg = stateless_init_torch_distributed_process_group(master_address, master_port, rank, world_size, backend="gloo")
    logger.info("vLLM weight-sync backend: gloo (CPU staging)")
    return _GlooBroadcastCommunicator(pg, device)


def _xpu_physical_id(index):
    """UUID and PCI address of visible XPU ``index``. torch exposes no PCI field; Intel's Level Zero
    UUID carries the PCI domain/bus/device/function, accepted only if sysfs has that device."""
    import os

    import torch

    uuid = str(torch.xpu.get_device_properties(index).uuid)
    raw = bytes.fromhex(uuid.replace("-", ""))
    bdf = f"{int.from_bytes(raw[6:8], 'little'):04x}:{raw[8]:02x}:{raw[9]:02x}.{raw[10]:x}"
    return uuid, bdf if os.path.exists(f"/sys/bus/pci/devices/{bdf}") else None


def _bitsandbytes_xpu_tables():
    """(XPU index, bytes) of the lookup tables bitsandbytes creates at import (NF4/FP4 in
    bitsandbytes/backends/utils.py, observed with 0.50.2) when they sit on an XPU, else None.
    They land on whichever XPU is current at import, and transformers imports bitsandbytes
    before a Ray actor can select its device."""
    import sys

    utils = sys.modules.get("bitsandbytes.backends.utils")
    tables = [getattr(utils, name, None) for name in ("_NF4_QUANT_TABLE", "_FP4_QUANT_TABLE")] if utils else []
    tables = [t for t in tables if t is not None and t.device.type == "xpu"]
    if not tables or len({t.device.index for t in tables}) != 1:
        return None
    return tables[0].device.index, sum(t.numel() * t.element_size() for t in tables)


def xpu_identity(role, **extra):
    """This process's XPU placement by physical identity (UUID / PCI address), never by xpu:N alone."""
    import os
    import socket

    import torch

    visible = [_xpu_physical_id(i) for i in range(torch.xpu.device_count())]
    current = torch.xpu.current_device()
    return {
        "role": role,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK"),
        "ONEAPI_DEVICE_SELECTOR": os.environ.get("ONEAPI_DEVICE_SELECTOR"),
        "device_count": len(visible),
        "current_device": current,
        "name": torch.xpu.get_device_name(current),
        "uuid": visible[current][0],
        "pci": visible[current][1],
        "visible_uuids": [uuid for uuid, _ in visible],
        "visible_pci": [pci for _, pci in visible],
        # caching-allocator bytes per visible XPU: everything must sit on the selected one
        "reserved_bytes": [torch.xpu.memory_reserved(i) for i in range(len(visible))],
        "bitsandbytes_tables": _bitsandbytes_xpu_tables(),
        **extra,
    }


def check_xccl_participants(participants, reference_uuids, colocated=False, xccl=True, reference_pci=None):
    """Raise XcclUnavailableError unless XCCL weight sync is safe for these participants.

    Each participant is an xpu_identity() dict plus "assigned_xpu", its Ray-assigned XPU: an index into
    its host's XPU enumeration. ``reference_uuids`` / ``reference_pci`` are that enumeration on the first
    participant's (the trainer's) host; a participant on another host is checked against its own list.
    ``xccl=False`` checks placement only (gloo): Ray assignment and memory, not visibility or sharing.
    """
    errors = []
    for p in participants:
        who = f"{p['role']} (weight-sync rank {p['weight_sync_rank']}, pid {p['pid']})"
        assigned = p["assigned_xpu"]
        uuids, pcis = (
            (reference_uuids, reference_pci)
            if p["host"] == participants[0]["host"]
            else (p["visible_uuids"], p["visible_pci"])
        )
        expected = uuids[assigned] if 0 <= assigned < len(uuids) else None
        if p["uuid"] != expected:
            expected_pci = pcis[assigned] if pcis and 0 <= assigned < len(pcis) else "?"
            errors.append(
                f"{who} selected xpu:{p['current_device']} = {p['uuid']} ({p['pci']}), but Ray assigned XPU "
                f"{assigned} = {expected} ({expected_pci})"
            )
        # vLLM's own process groups should use the worker's XPU too. vLLM 0.31 maps device_ids in its XPU worker
        # but not in GroupCoordinator; with one worker per engine the groups have no device communicator, so the
        # mismatch is reported, not fatal. With more workers it would misplace vLLM's own transfers.
        stray_groups = [d for d in p.get("group_devices", ()) if d != f"xpu:{p['current_device']}"]
        if stray_groups:
            message = f"{who} runs on xpu:{p['current_device']}, but vLLM's process groups use {stray_groups}"
            if p.get("engine_world_size", 1) > 1:
                errors.append(message)
            else:
                logger.warning(
                    message + " (vLLM GroupCoordinator does not map device_ids on XPU; unused with one worker)"
                )
        other = [i for i, n in enumerate(p["reserved_bytes"]) if n and i != p["current_device"]]
        tables = p["bitsandbytes_tables"]
        if tables and other == [tables[0]] and p["reserved_bytes"][tables[0]] <= 2 << 20:
            # one allocator block holding only bitsandbytes' import-time tables
            logger.warning(
                f"{who}: {p['reserved_bytes'][tables[0]]} bytes on XPU {p['visible_uuids'][tables[0]]} hold "
                f"bitsandbytes' import-time lookup tables ({tables[1]} bytes), not its own data"
            )
        elif other:
            errors.append(
                f"{who} has memory on XPU(s) {[p['visible_uuids'][i] for i in other]}, not only on its own {p['uuid']}"
            )
    for p in participants:
        # oneCCL needs every communicator XPU of the same host visible; XPUs of other hosts are reached over the network
        same_host = {q["uuid"] for q in participants if q["host"] == p["host"]} if xccl else set()
        missing = same_host - set(p["visible_uuids"])
        if missing:
            errors.append(
                f"{p['role']} (weight-sync rank {p['weight_sync_rank']}, pid {p['pid']}) cannot see communicator "
                f"XPU(s) {sorted(missing)} (ZE_AFFINITY_MASK={p['ZE_AFFINITY_MASK']}, "
                f"ONEAPI_DEVICE_SELECTOR={p['ONEAPI_DEVICE_SELECTOR']}); oneCCL needs every communicator XPU "
                "visible in every process"
            )
    if xccl and len({p["uuid"] for p in participants}) < len(participants) and not colocated:
        errors.append("two weight-sync ranks run on the same XPU, but vLLM is not colocated with the trainer")
    if errors:
        raise XcclUnavailableError("XCCL weight-sync placement check failed:\n  " + "\n  ".join(errors))


_XCCL_WATCHDOG_KV_NAMESPACE = "openrlhf_xccl_watchdog"


def xccl_watchdog_markers(attempt):
    """Watchdog failure markers of launch attempt ``attempt``; markers of other attempts never match."""
    import json

    from ray.experimental import internal_kv

    if not attempt:
        return []
    keys = internal_kv._internal_kv_list(f"{attempt}/", namespace=_XCCL_WATCHDOG_KV_NAMESPACE)
    return [json.loads(internal_kv._internal_kv_get(k, namespace=_XCCL_WATCHDOG_KV_NAMESPACE)) for k in keys]


class ProcessWatchdog:
    """Ends the process if a block does not finish in time. A hung XCCL collective cannot be
    cancelled (ProcessGroupXCCL's timeout does not fire) and its device queue is untrusted, so the
    only safe outcome is to stop: Ray then reports the actor's death instead of waiting forever.

    ``marker``: context of the structured failure marker written before exiting (launch attempt,
    stage, actor, rank, device UUID and PCI BDF)."""

    def __init__(self, timeout, what, marker=None):
        self.timeout = timeout
        self.what = what
        self.phase = "start"
        self.marker = marker
        self._timer = None

    def __enter__(self):
        self.restart(self.phase)
        return self

    def restart(self, phase):
        """Start the timeout over for ``phase``: restarted per step, each step (not the block) must finish in time."""
        import threading

        if self._timer:
            self._timer.cancel()
        self.phase = phase
        self._timer = threading.Timer(self.timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def __exit__(self, *exc):
        self._timer.cancel()
        return False

    def _expire(self):
        import json
        import os
        import sys

        print(
            f"FATAL: {self.what} did not finish within {self.timeout}s (phase: {self.phase}); "
            "stopping this process - device state is untrusted, no fallback is attempted.",
            file=sys.stderr,
            flush=True,
        )
        if self.marker is not None:
            # Structured marker, keyed by launch attempt in Ray's KV store: the launcher falls back only
            # for a death it can attribute to this watchdog (needs a Ray actor process, e.g. the trainer).
            marker = dict(self.marker, failure="xccl_watchdog_timeout", phase=self.phase, what=self.what)
            marker.update(timeout_s=self.timeout, pid=os.getpid())
            text = json.dumps(marker, default=str)
            print("XCCL-WATCHDOG-MARKER " + text, file=sys.stderr, flush=True)
            try:
                import ray
                from ray.experimental import internal_kv

                if marker.get("attempt") and ray.is_initialized():
                    key = f"{marker['attempt']}/{marker.get('rank')}/{marker['pid']}"
                    internal_kv._internal_kv_put(key, text, namespace=_XCCL_WATCHDOG_KV_NAMESPACE)
            except Exception:  # the process exits next; the printed marker remains
                pass
        os._exit(1)


def xccl_preflight(group, rank, device, identity="", timeout=120, marker=None):
    """Deterministic test broadcasts from rank 0 before any real weight: 4 KiB, 16 MiB, 128 MiB
    (large buffers are where oneCCL IPC faults show up). Receivers start poison-filled; every rank
    checks bit-exact after synchronize and again after a short delay. Returns per-size results;
    ``identity`` (rank, UUID, PCI address) is included in every failure. ``marker``: watchdog
    failure-marker context (see ProcessWatchdog)."""
    import time

    import torch

    def synchronize():
        if device.type == "xpu":
            torch.xpu.synchronize(device)

    results = []
    with ProcessWatchdog(timeout, f"XCCL preflight on rank {rank} [{identity}]", marker=marker) as watchdog:
        for nbytes in (4 << 10, 16 << 20, 128 << 20):
            n = nbytes // 2  # bf16 elements; 0..250 repeated, built without int64 temporaries
            expected = torch.arange(251, dtype=torch.bfloat16, device=device).repeat(n // 251 + 1)[:n]
            tensor = expected.clone() if rank == 0 else torch.full_like(expected, -12345.0)
            watchdog.phase = f"{nbytes} B broadcast"
            group.broadcast(tensor, src=0)
            synchronize()
            for phase in ("after synchronize", "0.2 s later"):
                if phase != "after synchronize":
                    time.sleep(0.2)
                    synchronize()
                bad = (tensor != expected).nonzero()
                if len(bad):
                    i = int(bad[0])
                    results.append(
                        {
                            "bytes": nbytes,
                            "ok": False,
                            "phase": phase,
                            "identity": identity,
                            "first_mismatch": f"element {i}: {tensor[i].item()} != {expected[i].item()}",
                        }
                    )
                    break
            else:
                results.append({"bytes": nbytes, "ok": True})
    return results
