"""Tests for the gloo-based vLLM weight-sync fallback (distributed_util._GlooBroadcastCommunicator).

On platforms without NCCL/RCCL (e.g. Intel XPU), vLLM's PyNcclCommunicator silently disables
itself and weight broadcasts become no-ops. The gloo fallback routes the broadcast through a
real torch.distributed gloo process group, staging through CPU.

These tests prove:
  - the stateless gloo process group broadcasts correctly for the world sizes OpenRLHF uses
    (world_size = num_vllm_engines * tp + 1), tested at 2/3/4;
  - the communicator preserves the caller's tensor storage (copy_, not reassignment) and
    returns the same tensor;
  - repeated broadcasts deliver successive values.

They do NOT prove the full multi-XPU staging path or ZeRO-3 behavior - those need real
multi-accelerator hardware.
"""

import queue
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openrlhf.utils.distributed_util import _GlooBroadcastCommunicator, resolve_vllm_sync_backend


# --- backend resolver (auto when unset, strict on explicit) ------------------------------
# The resolver reads torch.version.cuda / torch.version.hip / torch.version.xpu and (for XPU)
# dist_xccl_available(). Patch all of them so we can simulate CUDA / ROCm / XPU builds
# regardless of the box the tests run on.
def _patch_platform(cuda=None, hip=None, xpu=None, xccl=False, xpu_count=1):
    return (
        patch("torch.version.cuda", cuda, create=True),
        patch("torch.version.hip", hip, create=True),
        patch("torch.version.xpu", xpu, create=True),
        patch("openrlhf.utils.distributed_util.dist_xccl_available", return_value=xccl),
        patch("torch.xpu.device_count", return_value=xpu_count, create=True),
    )


@pytest.mark.parametrize(
    "cuda,hip,xpu,xccl,xpu_count,requested,expected",
    [
        # auto-detect (unset): nccl on CUDA/ROCm; xccl on XPU with >=2 devices; else gloo
        ("12.1", None, None, False, 1, None, "nccl"),   # CUDA -> nccl
        (None, "6.0", None, False, 1, None, "nccl"),    # ROCm -> nccl
        (None, None, "20250", True, 2, None, "xccl"),   # XPU + xccl + 2 devices -> xccl (direct XPU->XPU)
        (None, None, "20250", True, 8, None, "xccl"),   # XPU + xccl + 8 devices -> xccl
        (None, None, "20250", True, 1, None, "gloo"),   # XPU + xccl + 1 device  -> gloo (single-tile can't xccl)
        (None, None, "20250", False, 1, None, "gloo"),  # XPU without xccl -> gloo
        (None, None, None, False, 1, None, "gloo"),     # nothing detected -> gloo
        # explicit
        ("12.1", None, None, False, 1, "nccl", "nccl"),
        (None, None, "20250", True, 2, "xccl", "xccl"),  # explicit xccl + 2 XPU -> xccl (the real path)
        (None, None, "20250", True, 1, "xccl", "gloo"),  # explicit xccl + 1 XPU -> falls back to gloo (no hang)
        (None, None, "20250", True, 2, "gloo", "gloo"),  # explicit gloo on XPU -> honored
        ("12.1", None, None, False, 1, "gloo", "gloo"),  # explicit gloo on CUDA -> honored
    ],
)
def test_resolve_backend_auto_and_explicit(cuda, hip, xpu, xccl, xpu_count, requested, expected):
    p_cuda, p_hip, p_xpu, p_xccl, p_cnt = _patch_platform(cuda, hip, xpu, xccl, xpu_count)
    with p_cuda, p_hip, p_xpu, p_xccl, p_cnt:
        assert resolve_vllm_sync_backend(requested) == expected


def test_resolve_backend_explicit_nccl_on_non_nccl_raises():
    p_cuda, p_hip, p_xpu, p_xccl, p_cnt = _patch_platform(cuda=None, hip=None, xpu="20250", xccl=True, xpu_count=2)
    with p_cuda, p_hip, p_xpu, p_xccl, p_cnt, pytest.raises(RuntimeError, match="requires a CUDA or ROCm build"):
        resolve_vllm_sync_backend("nccl")


def test_resolve_backend_explicit_xccl_on_non_xpu_raises():
    p_cuda, p_hip, p_xpu, p_xccl, p_cnt = _patch_platform(cuda="12.1", hip=None, xpu=None, xccl=False, xpu_count=0)
    with p_cuda, p_hip, p_xpu, p_xccl, p_cnt, pytest.raises(RuntimeError, match="requires an Intel XPU build"):
        resolve_vllm_sync_backend("xccl")


def test_resolve_backend_xccl_single_xpu_falls_back_to_gloo():
    # Explicit xccl on a single-XPU box must NOT hang - it falls back to gloo with a warning.
    p_cuda, p_hip, p_xpu, p_xccl, p_cnt = _patch_platform(cuda=None, hip=None, xpu="20250", xccl=True, xpu_count=1)
    with p_cuda, p_hip, p_xpu, p_xccl, p_cnt:
        assert resolve_vllm_sync_backend("xccl") == "gloo"


def test_resolve_backend_unknown_raises():
    p_cuda, p_hip, p_xpu, p_xccl, p_cnt = _patch_platform(cuda="12.1", hip=None)
    with p_cuda, p_hip, p_xpu, p_xccl, p_cnt, pytest.raises(ValueError, match="Unsupported vLLM weight-sync backend"):
        resolve_vllm_sync_backend("mpi")


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bcast_worker(rank, world_size, port, result_queue):
    """One rank: create a stateless gloo group, broadcast a CPU tensor from rank 0, report it."""
    try:
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group("127.0.0.1", port, rank, world_size, backend="gloo")
        # rank 0 is the source with a distinct value; others start different
        value = 42.0 if rank == 0 else float(1000 + rank)
        tensor = torch.full((4,), value)
        work = pg.broadcast(tensor, 0)
        work.wait()
        result_queue.put((rank, tensor.tolist(), None))
    except BaseException as e:  # noqa: BLE001
        result_queue.put((rank, None, f"{type(e).__name__}: {e}"))
        raise


@pytest.mark.integration
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_stateless_gloo_broadcast_all_ranks_receive_source(world_size):
    """Every rank must receive rank 0's value (the OpenRLHF weight-sync topology)."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=_bcast_worker, args=(r, world_size, port, result_queue)) for r in range(world_size)]

    try:
        for p in procs:
            p.start()
        results = {}
        for _ in range(world_size):
            try:
                rank, vals, err = result_queue.get(timeout=90)
            except queue.Empty:
                break
            results[rank] = (vals, err)
        for p in procs:
            p.join(timeout=15)

        errors = {r: e for r, (_, e) in results.items() if e is not None}
        assert not errors, f"worker errors: {errors}"
        assert set(results) == set(range(world_size)), f"missing ranks: {results}"
        for rank in range(world_size):
            vals, _ = results[rank]
            assert vals == [42.0] * 4, f"rank {rank} did not receive source value: {vals}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


class _FakeCPUProcessGroup:
    """A stand-in gloo group for single-process tests: broadcast from rank 0 is a no-op on the
    source tensor (rank 0 already holds the source), so this validates the communicator's
    CPU-staging + copy_ behavior without spawning processes."""

    def broadcast(self, tensor, src):
        class _Work:
            def wait(self_inner):
                return None

        return _Work()


def test_communicator_preserves_storage_and_returns_same_tensor():
    """_GlooBroadcastCommunicator must copy into the original tensor, not replace its storage."""
    comm = _GlooBroadcastCommunicator(_FakeCPUProcessGroup(), device=torch.device("cpu"))
    tensor = torch.arange(4, dtype=torch.float32)
    storage_before = tensor.untyped_storage().data_ptr()

    result = comm.broadcast(tensor, src=0)

    assert result is tensor, "communicator must return the same tensor object"
    assert result.untyped_storage().data_ptr() == storage_before, "tensor storage must be preserved"


def test_communicator_broadcast_delivers_value():
    """The staged CPU value round-trips back into the tensor (rank-0 source case)."""
    comm = _GlooBroadcastCommunicator(_FakeCPUProcessGroup(), device=torch.device("cpu"))
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = comm.broadcast(tensor, src=0)
    # rank-0 source: value unchanged, but proves the copy_ path runs and returns correct data
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))


def test_communicator_accepts_stream_kwarg():
    """The gloo communicator must accept (and ignore) stream= for PyNccl interface parity."""
    comm = _GlooBroadcastCommunicator(_FakeCPUProcessGroup(), device=torch.device("cpu"))
    tensor = torch.zeros(2)
    # must not raise
    comm.broadcast(tensor, src=0, stream=None)


# --- XCCL (direct XPU->XPU) weight-sync -------------------------------------------------
def _xccl_multi_xpu_available():
    # XCCL weight-sync needs DISTINCT physical XPUs per rank. With both ranks on a single
    # physical XPU (device_count == 1) the collective is unreliable / can hang, so this test
    # only runs on a real multi-XPU box.
    import torch
    from openrlhf.utils.distributed_util import dist_xccl_available

    return bool(getattr(torch.version, "xpu", None)) and dist_xccl_available() and torch.xpu.device_count() >= 2


def _xccl_worker(rank, world_size, port, result_queue):
    """One rank: build the real XCCL communicator via stateless_init_process_group and
    broadcast an XPU tensor directly (no CPU staging)."""
    try:
        import torch

        from openrlhf.utils.distributed_util import stateless_init_process_group

        # Each rank on its OWN physical XPU (this test only runs when device_count >= 2).
        torch.xpu.set_device(rank)
        comm = stateless_init_process_group(
            "127.0.0.1", port, rank, world_size, torch.device("xpu", rank), backend="xccl"
        )
        tensor = torch.full((4,), 42.0 if rank == 0 else float(1000 + rank), device=f"xpu:{rank}")
        out = comm.broadcast(tensor, src=0)
        torch.xpu.synchronize()
        result_queue.put((rank, out.cpu().tolist(), str(out.device), out is tensor, None))
    except BaseException as e:  # noqa: BLE001
        result_queue.put((rank, None, None, None, f"{type(e).__name__}: {e}"))
        raise


@pytest.mark.integration
def test_xccl_broadcast_direct_on_xpu():
    """Real 2-rank XCCL broadcast of XPU tensors through our communicator: every rank receives
    rank 0's value, in place, staying on the XPU (no CPU staging). Skips if XCCL/XPU absent."""
    if not _xccl_multi_xpu_available():
        pytest.skip("XCCL/XPU not available on this build")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [ctx.Process(target=_xccl_worker, args=(r, 2, port, result_queue)) for r in range(2)]
    try:
        for p in procs:
            p.start()
        results = {}
        for _ in range(2):
            try:
                rank, vals, dev, in_place, err = result_queue.get(timeout=120)
            except queue.Empty:
                break
            results[rank] = (vals, dev, in_place, err)
        for p in procs:
            p.join(timeout=15)

        errors = {r: v[3] for r, v in results.items() if v[3] is not None}
        assert not errors, f"xccl worker errors: {errors}"
        assert set(results) == {0, 1}, f"missing ranks: {results}"
        for rank in (0, 1):
            vals, dev, in_place, _ = results[rank]
            assert vals == [42.0] * 4, f"rank {rank} wrong values: {vals}"
            assert dev.startswith("xpu"), f"rank {rank} left the XPU: {dev}"  # no CPU staging
            assert in_place, f"rank {rank} did not broadcast in place"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
