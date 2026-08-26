import asyncio
import os
from copy import deepcopy
from typing import Any, List, Optional

import ray
from packaging import version
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .utils import get_bundle_indices, ray_noset_visible_devices

# Keep runtime-sensitive imports (vllm, openrlhf.utils.agent) out of module scope so accelerator
# visibility can be configured before any dependency initializes the device runtime inside a
# Ray actor. These are imported locally in the methods that use them, after device pinning.


def _resolve_device_control_env_var():
    """Env var that controls device visibility: XPU -> ZE_AFFINITY_MASK, CUDA/ROCm ->
    CUDA_VISIBLE_DEVICES. Reads torch.version, not vllm.platforms, to avoid initializing the
    runtime before the device is pinned."""
    import torch

    if getattr(torch.version, "xpu", None):
        return "ZE_AFFINITY_MASK"
    if getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
        return "CUDA_VISIBLE_DEVICES"
    return None


def _load_agent_executor(agent_func_path: str):
    assert agent_func_path.endswith(".py"), "Agent path must be a Python file"

    import importlib.util

    # Local import: keeps this Ray actor module from pulling runtime-sensitive deps at import.
    from openrlhf.utils.agent import AgentExecutorBase

    spec = importlib.util.spec_from_file_location("agent_module", agent_func_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)

    assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
    agent_executor_cls = agent_module.AgentExecutor
    assert issubclass(agent_executor_cls, AgentExecutorBase), "AgentExecutor must inherit from AgentExecutorBase"
    return agent_executor_cls()


@ray.remote
class RolloutRayActor:
    """Async vLLM-backed actor that exposes generation utilities."""

    async def __init__(
        self,
        *args,
        bundle_indices: list = None,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        mm_pad_token_ids: Optional[set] = None,
        **kwargs,
    ):
        # Configure device visibility before importing vLLM or other runtime-sensitive dependencies.
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=kwargs.pop("num_gpus"),
        )

        # Import vLLM only after device visibility has been configured, before
        # any dependency has an opportunity to initialize the accelerator runtime

        import vllm

        self._configure_vllm_env(version, vllm, kwargs.pop("full_determinism", False))

        # Local import, deferred for the same reason as vllm above.
        from openrlhf.utils.agent import SingleTurnAgentExecutor

        if agent_func_path:
            self.executor = _load_agent_executor(agent_func_path)
        else:
            self.executor = SingleTurnAgentExecutor(remote_rm_url)

        self.kwargs = kwargs

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

        # Used by generate() to collapse pre-expanded image/video pad tokens
        # before passing to vLLM (avoids double-expansion).
        self._mm_pad_token_ids = mm_pad_token_ids

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        # Device-visibility env var for the active platform (NVIDIA -> CUDA_VISIBLE_DEVICES,
        # Intel XPU -> ZE_AFFINITY_MASK). Read from torch.version, not vllm.platforms, since
        # importing vllm.platforms would initialize the runtime before pinning.
        device_control_env_var = _resolve_device_control_env_var()
        if not device_control_env_var:
            raise RuntimeError(
                "Could not determine the device-control environment variable for the active accelerator"
            )

        if backend == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray, so
            # vLLM's ray executor is free to assign devices to its child workers.
            # ONEAPI_DEVICE_SELECTOR is cleared too so an inherited Intel selector
            # is not passed down to child workers.
            for env_var in (
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ONEAPI_DEVICE_SELECTOR",
                device_control_env_var,
            ):
                os.environ.pop(env_var, None)
        else:
            # Set the platform's device-visibility var to the Ray-assigned accelerator when
            # either RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set (Ray does not set it) or
            # the var is simply absent - e.g. Intel XPU, where Ray sets CUDA_VISIBLE_DEVICES
            # but not ZE_AFFINITY_MASK, the var vLLM's XPU platform actually reads.
            #
            # ZE_AFFINITY_MASK must be set before the Level-Zero runtime initializes; this runs
            # before any accelerator API call. The value is the Ray-assigned accelerator ID;
            # how ZE_AFFINITY_MASK maps it (root device vs. tile) depends on the L0 hierarchy.
            if ray_noset_visible_devices() or device_control_env_var not in os.environ:
                gpu_ids = ray.get_gpu_ids()
                if not gpu_ids:
                    raise RuntimeError("Ray did not assign an accelerator to the vLLM actor")
                # ONEAPI_DEVICE_SELECTOR and ZE_AFFINITY_MASK intersect in Level-Zero; if Ray
                # set the selector, keeping it can yield 0 visible devices. Clear before masking.
                if device_control_env_var == "ZE_AFFINITY_MASK":
                    os.environ.pop("ONEAPI_DEVICE_SELECTOR", None)
                os.environ[device_control_env_var] = str(gpu_ids[0])

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

    def _configure_vllm_env(self, version, vllm, full_determinism: bool):
        assert version.parse(vllm.__version__) > version.parse(
            "0.8.5"
        ), "Streaming VLLM version must be greater than 0.8.5"

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

        os.environ["VLLM_USE_V1"] = "1"

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache),
        )

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def pause_generation(self):
        await self.llm.pause_generation(mode="keep")

    async def resume_generation(self):
        await self.llm.resume_generation()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self, tags=None):
        """Wake up the engine from sleep mode.

        Args:
            tags: Optional list of tags to selectively wake up.
                  Use ["weights"] to wake up only model weights (for weight sync).
                  Use ["kv_cache"] to wake up only KV cache (after weight sync).
                  Use None to wake up everything.
        """
        if tags is None:
            tags = ["weights", "kv_cache"]
        for tag in tags:
            await self.llm.wake_up(tags=[tag])

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None):
        """Token-level generation for rollout executors."""
        if multi_modal_data and self._mm_pad_token_ids:
            # Collapse consecutive image/video pad tokens to a single
            # placeholder so vLLM's multimodal processor expands them
            # exactly once (avoids double-expansion).
            from openrlhf.utils.vlm_utils import dedup_media_tokens

            prompt_token_ids = dedup_media_tokens(prompt_token_ids, self._mm_pad_token_ids)

        # Lazy import - see module-top note: vllm is never imported at module level.
        from vllm.inputs import TokensPrompt
        from vllm.utils import random_uuid

        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data

        generator = self.llm.generate(
            prompt,
            deepcopy(sampling_params),
            request_id=random_uuid(),
        )

        final_output = None
        async for request_output in generator:
            final_output = request_output

        return final_output

    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in vLLM engine."""
        return self.llm.output_processor.get_num_unfinished_requests()

    async def generate_responses(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        num_samples: int = 1,
        images=None,
    ):
        """Generate N samples for a single prompt."""
        tasks = [
            self.executor.execute(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=self,
                images=images,
            )
            for _ in range(num_samples)
        ]
        return await asyncio.gather(*tasks)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    logprobs_mode=None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
    max_images_per_prompt: int = 0,
):
    """Spin up a set of vLLM Ray actors with consistent placement."""
    # Detect VLM pad token IDs once, shared across all engines.
    mm_pad_token_ids: set = set()
    if max_images_per_prompt > 0:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True)
        for attr in ("image_token_id", "video_token_id"):
            tid = getattr(processor, attr, None)
            if tid is not None:
                mm_pad_token_ids.add(tid)
        del processor

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # allow two engines to share one GPU in hybrid mode
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_extension_cls": "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "full_determinism": full_determinism,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": 0.2 if use_hybrid_engine else 1,
            "enable_sleep_mode": vllm_enable_sleep,
        }

        actor_kwargs.update(
            {
                "agent_func_path": agent_func_path,
                "remote_rm_url": remote_rm_url,
                "mm_pad_token_ids": mm_pad_token_ids,
            }
        )

        if max_images_per_prompt > 0:
            actor_kwargs["limit_mm_per_prompt"] = {"image": max_images_per_prompt}

        if logprobs_mode:
            # Lazy import - see module-top note: vllm is never imported at module level.
            import vllm

            actor_kwargs["logprobs_mode"] = logprobs_mode
            actor_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            RolloutRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Call the same method on a list of engines and gather results."""
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
