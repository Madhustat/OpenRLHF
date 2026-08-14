"""Tests for the gloo-based vLLM weight-sync fallback (distributed_util._GlooBroadcastCommunicator).

On platforms without NCCL/RCCL (e.g. Intel XPU), vLLM's PyNcclCommunicator silently disables
itself and weight broadcasts become no-ops. The gloo fallback routes the broadcast through a
real torch.distributed gloo process group, staging through CPU.

Test levels
-----------
Unit tests (no spawned processes, no hardware required)
    - Backend resolver: auto-select and explicit values
    - Communicator contract: copy_ semantics, storage preservation, stream kwarg parity

Integration tests (real multi-process gloo, no GPU required)  [pytest.mark.integration]
    - Stateless gloo broadcast across world_size 2/3/4
    - Exact weight-equality: receiver gets precisely what sender sent (rtol=0, atol=0)
    - Receiver value changed from its initial state (not stale)
    - Repeated broadcasts deliver successive distinct values
    - ZeRO-style parameter flow: sender gathers parameter, stages to CPU, broadcasts;
      receiver updates its local copy in-place

Hardware tests (real XPU devices)  [pytest.mark.integration, skipped without hardware]
    - XCCL direct XPU-to-XPU broadcast

What these tests do NOT cover (requires real actor+vLLM E2E on hardware)
    - ZeRO-3 GatheredParameters lifecycle with DeepSpeed sharding
    - LoRA merge-before-broadcast correctness
    - vLLM load_weights() parameter name mapping
    - Multi-node weight sync
"""

import queue
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openrlhf.utils.distributed_util import _GlooBroadcastCommunicator, resolve_vllm_sync_backend


# --- backend resolver (auto when unset, strict on explicit) ------------------------------
# The resolver reads torch.version.cuda / torch.version.hip / torch.version.xpu, (for XPU)
# dist_xccl_available(), and the PHYSICAL XPU count. Patch all of them so we can simulate
# CUDA / ROCm / XPU builds regardless of the box the tests run on. xpu_count models the number
# of PHYSICAL XPUs on the host; both the sysfs probe (_physical_xpu_count) and the per-process
# torch.xpu.device_count() are patched to it so the resolver sees a consistent world (the
# resolver keys on max(physical, visible), so both must reflect the simulated count).
class _patch_count:
    """One context manager that patches BOTH the physical-XPU probe and torch.xpu.device_count,
    so _patch_platform keeps its 5-tuple shape (callers unpack 5) while the resolver's
    max(_physical_xpu_count(), device_count()) sees the simulated count consistently."""

    def __init__(self, xpu_count):
        self._physical = patch("openrlhf.utils.distributed_util._physical_xpu_count", return_value=xpu_count)
        self._visible = patch("torch.xpu.device_count", return_value=xpu_count, create=True)

    def __enter__(self):
        self._physical.__enter__()
        self._visible.__enter__()
        return self

    def __exit__(self, *exc):
        self._visible.__exit__(*exc)
        self._physical.__exit__(*exc)
        return False


def _patch_platform(cuda=None, hip=None, xpu=None, xccl=False, xpu_count=1):
    return (
        patch("torch.version.cuda", cuda, create=True),
        patch("torch.version.hip", hip, create=True),
        patch("torch.version.xpu", xpu, create=True),
        patch("openrlhf.utils.distributed_util.dist_xccl_available", return_value=xccl),
        _patch_count(xpu_count),
    )


@pytest.mark.parametrize(
    "cuda,hip,xpu,xccl,xpu_count,requested,expected",
    [
        # auto-detect (unset): nccl on CUDA/ROCm; gloo everywhere else. XPU auto stays gloo (the
        # safe, known-working default) EVEN with >=2 devices - xccl is opt-in only, because
        # torch 2.12.0+xpu segfaults in ProcessGroupXCCL on Battlemage/PCIe (torch-xpu-ops #4238).
        ("12.1", None, None, False, 1, None, "nccl"),   # CUDA -> nccl
        (None, "6.0", None, False, 1, None, "nccl"),    # ROCm -> nccl
        (None, None, "20250", True, 2, None, "gloo"),   # XPU + xccl + 2 devices -> gloo (xccl NOT auto)
        (None, None, "20250", True, 8, None, "gloo"),   # XPU + xccl + 8 devices -> gloo (xccl NOT auto)
        (None, None, "20250", True, 1, None, "gloo"),   # XPU + xccl + 1 device  -> gloo
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


# --- Integration tests: exact weight-equality across real processes ----------------------
#
# These tests spawn real subprocesses with a real gloo process group (no GPU required) and
# prove the numerical correctness guarantee that the upstream PR requires:
#
#   1. The receiver gets exactly the value the sender sent (rtol=0, atol=0).
#   2. The receiver's value actually changed from its initial state (not stale).
#   3. The communicator's copy_() path preserves tensor storage across a real network hop.
#   4. Repeated broadcasts deliver successive distinct values (no caching artefact).
#   5. A ZeRO-style parameter flow (gather -> stage to CPU -> broadcast -> in-place update)
#      produces exact equality with the gathered value.
#
# These are integration tests rather than unit tests because they use real inter-process
# communication. They do not require any accelerator hardware.
# -----------------------------------------------------------------------------------------


def _weight_equality_worker(rank, world_size, port, result_queue):
    """
    Simulates the OpenRLHF actor-to-vLLM weight-sync data flow:

        Rank 0 = actor (sender):
            - holds a parameter that was updated by an optimizer step
            - broadcasts it through the gloo communicator

        Rank 1..N = vLLM workers (receivers):
            - start with a different (stale) value
            - receive the broadcast in-place via copy_()
            - report before/after values and storage pointer

    The test asserts:
        receiver_after  == sender_value    (exact, rtol=0 atol=0)
        receiver_after  != receiver_before (value actually changed)
        storage pointer unchanged          (copy_() not param.data replacement)
    """
    try:
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, world_size, backend="gloo"
        )
        comm = _GlooBroadcastCommunicator(pg, device=torch.device("cpu"))

        if rank == 0:
            # Actor: value after an optimizer step (distinct from the receiver's stale value)
            param = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.float32)
            storage_ptr = param.untyped_storage().data_ptr()
            result = comm.broadcast(param, src=0)
            result_queue.put({
                "rank": rank,
                "sent": param.tolist(),
                "storage_preserved": result.untyped_storage().data_ptr() == storage_ptr,
                "result_is_same_object": result is param,
                "err": None,
            })
        else:
            # vLLM worker: stale value before sync
            param = torch.zeros(4, dtype=torch.float32)
            before = param.clone()
            storage_ptr = param.untyped_storage().data_ptr()
            result = comm.broadcast(param, src=0)
            result_queue.put({
                "rank": rank,
                "before": before.tolist(),
                "after": result.tolist(),
                "storage_preserved": result.untyped_storage().data_ptr() == storage_ptr,
                "result_is_same_object": result is param,
                "err": None,
            })
    except BaseException as e:  # noqa: BLE001
        result_queue.put({"rank": rank, "err": f"{type(e).__name__}: {e}"})
        raise


@pytest.mark.integration
@pytest.mark.parametrize("world_size", [2, 3])
def test_gloo_weight_sync_exact_equality(world_size):
    """
    Core correctness proof for the actor-to-vLLM gloo weight-sync path.

    Proves:
      1. Every receiver gets exactly the sender's value with rtol=0 and atol=0
         (gloo is a copy operation; there must be zero numerical error).
      2. The receiver's parameter changed from its initial value
         (the sync is not a no-op).
      3. The communicator uses copy_() — tensor storage pointer is unchanged
         (no DeepSpeed/ZeRO parameter storage replacement).
      4. The same tensor object is returned (interface parity with PyNcclCommunicator).

    world_size=2 models the minimal OpenRLHF deployment (1 actor + 1 vLLM engine).
    world_size=3 models 1 actor + 1 vLLM engine with tensor_parallel_size=2.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_weight_equality_worker, args=(r, world_size, port, result_queue))
        for r in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        results = {}
        for _ in range(world_size):
            try:
                r = result_queue.get(timeout=60)
                results[r["rank"]] = r
            except queue.Empty:
                break
        for p in procs:
            p.join(timeout=15)

        # All ranks must have reported
        assert set(results) == set(range(world_size)), f"missing ranks: {set(range(world_size)) - set(results)}"

        # Check for worker errors
        errors = {r: v["err"] for r, v in results.items() if v.get("err")}
        assert not errors, f"worker errors: {errors}"

        sender = results[0]
        sender_value = torch.tensor(sender["sent"])

        for rank in range(1, world_size):
            recv = results[rank]
            receiver_after = torch.tensor(recv["after"])
            receiver_before = torch.tensor(recv["before"])

            # 1. Exact equality — gloo is a copy, not a compute; zero tolerance is correct
            torch.testing.assert_close(
                receiver_after, sender_value, rtol=0, atol=0,
                msg=f"rank {rank}: receiver value does not exactly match sender"
            )

            # 2. Value actually changed — the sync was not a no-op
            assert not torch.equal(receiver_after, receiver_before), (
                f"rank {rank}: receiver value unchanged after sync (stale weights)"
            )

            # 3. Storage preserved — copy_() not param.data replacement
            assert recv["storage_preserved"], (
                f"rank {rank}: tensor storage pointer changed (copy_() contract violated)"
            )

            # 4. Same object returned — PyNcclCommunicator interface parity
            assert recv["result_is_same_object"], (
                f"rank {rank}: communicator returned a different tensor object"
            )
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


def _repeated_broadcast_worker(rank, world_size, port, result_queue, n_rounds=5):
    """
    Verifies that repeated broadcasts deliver successive distinct values with no
    caching artefact — each round the receiver gets the sender's current value,
    which changes every round.
    """
    try:
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, world_size, backend="gloo"
        )
        comm = _GlooBroadcastCommunicator(pg, device=torch.device("cpu"))
        received = []

        for step in range(n_rounds):
            if rank == 0:
                # Sender: value increases each step (simulates optimizer updates)
                param = torch.full((3,), float(step + 1), dtype=torch.float32)
            else:
                # Receiver: always starts from a different value
                param = torch.zeros(3, dtype=torch.float32)

            comm.broadcast(param, src=0)
            received.append(param.tolist())

        result_queue.put({"rank": rank, "received": received, "err": None})
    except BaseException as e:  # noqa: BLE001
        result_queue.put({"rank": rank, "err": f"{type(e).__name__}: {e}"})
        raise


@pytest.mark.integration
def test_gloo_repeated_broadcasts_deliver_successive_values():
    """
    Repeated broadcasts deliver successive distinct values.

    In OpenRLHF, broadcast_to_vllm() is called after every training step.
    This test confirms there is no caching artefact — each call delivers the
    sender's current value, not a previously cached result.
    """
    world_size = 2
    n_rounds = 5
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(
            target=_repeated_broadcast_worker,
            args=(r, world_size, port, result_queue, n_rounds)
        )
        for r in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        results = {}
        for _ in range(world_size):
            try:
                r = result_queue.get(timeout=60)
                results[r["rank"]] = r
            except queue.Empty:
                break
        for p in procs:
            p.join(timeout=15)

        errors = {r: v["err"] for r, v in results.items() if v.get("err")}
        assert not errors, f"worker errors: {errors}"

        sender_rounds = results[0]["received"]
        receiver_rounds = results[1]["received"]

        for step in range(n_rounds):
            expected = float(step + 1)
            # Sender held the right value
            assert sender_rounds[step] == [expected] * 3, (
                f"step {step}: sender held wrong value {sender_rounds[step]}"
            )
            # Receiver got exactly the sender's value for this step
            assert receiver_rounds[step] == [expected] * 3, (
                f"step {step}: receiver got {receiver_rounds[step]}, expected {[expected]*3}"
            )
            # Each step's value is distinct from the previous (no caching)
            if step > 0:
                assert receiver_rounds[step] != receiver_rounds[step - 1], (
                    f"step {step}: receiver got same value as step {step-1} (possible caching)"
                )
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


def _zero_style_param_worker(rank, world_size, port, result_queue):
    """
    Simulates the ZeRO parameter gather -> CPU stage -> gloo broadcast -> in-place copy
    flow that ppo_actor.py uses in broadcast_to_vllm().

    Sender (rank 0):
        - holds a parameter as a contiguous CPU tensor (post ZeRO gather)
        - broadcasts it via _GlooBroadcastCommunicator

    Receiver (rank 1):
        - holds a zero-initialised buffer matching the parameter shape
        - receives the broadcast in-place
        - reports exact equality with the sender's value
    """
    try:
        from vllm.distributed.utils import stateless_init_torch_distributed_process_group

        pg = stateless_init_torch_distributed_process_group(
            "127.0.0.1", port, rank, world_size, backend="gloo"
        )
        comm = _GlooBroadcastCommunicator(pg, device=torch.device("cpu"))

        # Simulate a realistic model parameter shape (e.g. a small projection weight)
        param_shape = (64, 32)

        if rank == 0:
            # Post-optimizer-step parameter: non-trivial values
            torch.manual_seed(42)
            param = torch.randn(param_shape)
            param_value = param.clone()
            comm.broadcast(param, src=0)
            result_queue.put({"rank": rank, "value": param_value.tolist(), "err": None})
        else:
            # vLLM worker: empty buffer, will be filled by broadcast
            param = torch.zeros(param_shape)
            before_ptr = param.untyped_storage().data_ptr()
            comm.broadcast(param, src=0)
            result_queue.put({
                "rank": rank,
                "value": param.tolist(),
                "storage_unchanged": param.untyped_storage().data_ptr() == before_ptr,
                "err": None,
            })
    except BaseException as e:  # noqa: BLE001
        result_queue.put({"rank": rank, "err": f"{type(e).__name__}: {e}"})
        raise


@pytest.mark.integration
def test_gloo_zero_style_parameter_broadcast_exact_equality():
    """
    Proves exact weight equality through the ZeRO-style parameter flow.

    In ppo_actor.py, broadcast_to_vllm() gathers each parameter (for ZeRO-3),
    then calls _GlooBroadcastCommunicator.broadcast(param.data, src=0).

    This test simulates that flow with a realistic parameter shape (64x32) and
    asserts that the receiver's buffer matches the sender's gathered value exactly
    (rtol=0, atol=0) and that the receiver's storage pointer is unchanged.

    This is the closest approximation to the real broadcast_to_vllm() data path
    that can be tested without DeepSpeed, Ray, or vLLM.
    """
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_zero_style_param_worker, args=(r, world_size, port, result_queue))
        for r in range(world_size)
    ]
    try:
        for p in procs:
            p.start()
        results = {}
        for _ in range(world_size):
            try:
                r = result_queue.get(timeout=60)
                results[r["rank"]] = r
            except queue.Empty:
                break
        for p in procs:
            p.join(timeout=15)

        errors = {r: v["err"] for r, v in results.items() if v.get("err")}
        assert not errors, f"worker errors: {errors}"

        sender_value = torch.tensor(results[0]["value"])
        receiver_value = torch.tensor(results[1]["value"])

        # Exact equality — this is the upstream PR's core correctness requirement
        torch.testing.assert_close(
            receiver_value, sender_value, rtol=0, atol=0,
            msg="ZeRO-style parameter broadcast: receiver does not exactly match sender"
        )

        # Storage pointer unchanged — copy_() contract
        assert results[1]["storage_unchanged"], (
            "ZeRO-style parameter broadcast: receiver storage pointer changed (copy_() violated)"
        )
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


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
