import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openrlhf.utils import distributed_util
from openrlhf.utils.distributed_util import _GlooBroadcastCommunicator, resolve_vllm_sync_backend


def _platform(monkeypatch, kind, torch_version="2.14.0", xccl=True):
    monkeypatch.setattr(torch.version, "cuda", "12.8" if kind == "cuda" else None, raising=False)
    monkeypatch.setattr(torch.version, "hip", "6.4" if kind == "rocm" else None, raising=False)
    monkeypatch.setattr(torch.version, "xpu", "20260101" if kind == "xpu" else None, raising=False)
    monkeypatch.setattr(torch.distributed, "is_xccl_available", lambda: xccl, raising=False)
    monkeypatch.setattr(torch, "__version__", torch_version)


@pytest.mark.parametrize(
    "kind, colocated, expected",
    [
        ("cuda", False, "nccl"),
        ("cuda", True, "nccl"),
        ("rocm", False, "nccl"),
        ("xpu", False, "xccl"),
        ("xpu", True, "gloo"),
        ("cpu", False, "gloo"),
    ],
)
def test_auto_backend(monkeypatch, kind, colocated, expected):
    _platform(monkeypatch, kind)
    assert resolve_vllm_sync_backend("auto", colocated=colocated) == expected


@pytest.mark.parametrize("torch_version, xccl", [("2.12.0", True), ("2.14.0", False)])
def test_auto_uses_gloo_when_xccl_is_unusable(monkeypatch, torch_version, xccl):
    _platform(monkeypatch, "xpu", torch_version, xccl)
    assert resolve_vllm_sync_backend("auto", colocated=False) == "gloo"


@pytest.mark.parametrize("kind", ["cuda", "rocm", "xpu", "cpu"])
def test_explicit_gloo(monkeypatch, kind):
    _platform(monkeypatch, kind)
    assert resolve_vllm_sync_backend("gloo", colocated=False) == "gloo"


@pytest.mark.parametrize(
    "kind, backend, torch_version, message",
    [
        ("xpu", "nccl", "2.14.0", "nccl needs CUDA or ROCm"),
        ("cuda", "xccl", "2.14.0", "xccl needs an Intel XPU build of torch >= 2.13"),
        ("xpu", "xccl", "2.12.0", "xccl needs an Intel XPU build of torch >= 2.13"),
    ],
)
def test_explicit_backend_the_platform_cannot_run(monkeypatch, kind, backend, torch_version, message):
    _platform(monkeypatch, kind, torch_version)
    with pytest.raises(ValueError, match=message):
        resolve_vllm_sync_backend(backend, colocated=False)


def test_explicit_xccl_when_colocated_warns(monkeypatch):
    _platform(monkeypatch, "xpu")
    warnings = []
    monkeypatch.setattr(distributed_util.logger, "warning", warnings.append)
    assert resolve_vllm_sync_backend("xccl", colocated=True) == "xccl"
    assert len(warnings) == 1 and "experimental" in warnings[0]


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _gloo_broadcast(rank, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    communicator = _GlooBroadcastCommunicator(dist.group.WORLD)
    sent = torch.arange(1024, dtype=torch.bfloat16)
    tensor = sent.clone() if rank == 0 else torch.zeros_like(sent)
    storage = tensor.untyped_storage().data_ptr()
    received = communicator.broadcast(tensor, src=0, stream=None)
    assert received is tensor
    assert tensor.untyped_storage().data_ptr() == storage
    assert torch.equal(tensor, sent)
    dist.destroy_process_group()


def test_gloo_broadcast_fills_the_original_tensor():
    mp.spawn(_gloo_broadcast, args=(_free_port(),), nprocs=2)


def _xccl_broadcast(rank, port):
    device = torch.device("xpu", rank)
    communicator = distributed_util.stateless_init_process_group("127.0.0.1", port, rank, 2, device, backend="xccl")
    sent = torch.arange(1024, dtype=torch.bfloat16, device=device)
    tensor = sent.clone() if rank == 0 else torch.zeros_like(sent)
    received = communicator.broadcast(tensor, src=0)
    torch.xpu.synchronize(device)
    assert received is tensor
    assert torch.equal(tensor, sent)


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count() >= 2),
    reason="needs two Intel XPUs",
)
def test_xccl_broadcast_between_two_xpus():
    mp.spawn(_xccl_broadcast, args=(_free_port(),), nprocs=2)
