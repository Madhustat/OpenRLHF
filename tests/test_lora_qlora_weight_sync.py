"""Regression tests for LoRA/QLoRA actor-to-vLLM weight synchronization.

The tests exercise ``ActorPPOTrainer.broadcast_to_vllm`` with a recording
process group at the vLLM transport boundary. They verify the parameter name,
dtype, shape, and exact tensor value sent by the actor while avoiding a Ray
cluster or live vLLM engine.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

peft = pytest.importorskip("peft")
pytest.importorskip("deepspeed")
pytest.importorskip("ray")
pytest.importorskip("vllm")

ppo_actor = pytest.importorskip("openrlhf.trainer.ray.ppo_actor")


class _TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


class _RecordingProcessGroup:
    def __init__(self, engine):
        self.engine = engine

    def broadcast(self, tensor, src, stream):
        assert src == 0
        call = self.engine.update_weight.remote.call_args
        assert call is not None
        assert tensor.dtype == call.kwargs["dtype"]
        assert tensor.shape == call.kwargs["shape"]
        self.engine.received[call.args[0]] = tensor.detach().clone()


def _make_trainer(model):
    engine = SimpleNamespace(update_weight=MagicMock(), received={})
    trainer = ppo_actor.ActorPPOTrainer.__new__(ppo_actor.ActorPPOTrainer)
    trainer.actor = SimpleNamespace(model=SimpleNamespace(module=model), is_vlm=False)
    trainer.strategy = SimpleNamespace(
        args=SimpleNamespace(
            vllm=SimpleNamespace(enable_prefix_caching=False, sync_with_ray=False),
            ds=SimpleNamespace(zero_stage=2, tensor_parallel_size=1),
        )
    )
    trainer.vllm_engines = [engine]
    trainer._model_update_group = _RecordingProcessGroup(engine)
    trainer.use_cuda_ipc = False
    return trainer, engine


@pytest.fixture
def sync_runtime(monkeypatch):
    """Replace distributed coordination while retaining the real sync logic."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(ppo_actor.deepspeed.zero, "GatheredParameters", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(ppo_actor, "torch_dist_barrier_and_cuda_sync", lambda: None)
    monkeypatch.setattr(ppo_actor.ray, "get", lambda refs: refs)


def _make_qlora_model():
    bitsandbytes = pytest.importorskip("bitsandbytes")
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.is_loaded_in_4bit = True
    model.proj = bitsandbytes.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).cuda()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none"),
    )
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(
            torch.arange(128, device="cuda", dtype=torch.float32).reshape(2, 64) / 1000
        )
        layer.lora_B["default"].weight.copy_(
            torch.arange(128, device="cuda", dtype=torch.float32).reshape(64, 2) / 2000
        )
    return model


def test_lora_sync_sends_exact_effective_weight_and_restores_adapter(sync_runtime):
    model = peft.get_peft_model(
        _TinyActor(),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none"),
    )
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.base_layer.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
        layer.lora_A["default"].weight.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
        layer.lora_B["default"].weight.copy_(torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]]))
    base_weight = layer.base_layer.weight.detach().clone()
    expected = base_weight + layer.get_delta_weight("default").detach()

    trainer, engine = _make_trainer(model)
    trainer.broadcast_to_vllm()

    assert set(engine.received) == {"proj.weight"}
    assert engine.update_weight.remote.call_args.kwargs["empty_cache"]
    torch.testing.assert_close(engine.received["proj.weight"], expected, rtol=0, atol=0)
    assert not layer.merged
    tolerance = torch.finfo(base_weight.dtype).eps
    torch.testing.assert_close(layer.base_layer.weight, base_weight, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes 4-bit quantization requires CUDA")
def test_qlora_sync_sends_exact_materialized_bf16_weight_and_restores_adapter(sync_runtime):
    from bitsandbytes.functional import dequantize_4bit

    expected_model = _make_qlora_model()
    expected_layer = expected_model.base_model.model.proj
    expected_layer.merge(safe_merge=True)
    expected = dequantize_4bit(
        expected_layer.base_layer.weight.data,
        expected_layer.base_layer.weight.quant_state,
    ).to(torch.bfloat16)

    model = _make_qlora_model()
    layer = model.base_model.model.proj
    trainer, engine = _make_trainer(model)
    trainer.broadcast_to_vllm()

    assert set(engine.received) == {"proj.weight"}
    assert engine.update_weight.remote.call_args.kwargs["empty_cache"]
    assert engine.received["proj.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(engine.received["proj.weight"], expected, rtol=0, atol=0)
    assert not layer.merged
