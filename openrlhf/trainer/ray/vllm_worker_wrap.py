class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        import torch
        from openrlhf.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        self._sync_backend = backend
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
                backend=backend,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )
        if backend == "xccl" and not use_ray:
            from openrlhf.utils.distributed_util import xccl_preflight

            # Test broadcasts before any real weight; the trainer checks every rank's results.
            identity = f"vLLM rank {rank}, {self.device}, {self.xpu_identity()['uuid']}"
            return xccl_preflight(self._model_update_group, rank, self.device, identity)

    def xpu_identity(self):
        """This worker's XPU placement, for the trainer's XCCL placement check."""
        from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group, get_world_group

        from openrlhf.utils.distributed_util import xpu_identity

        return xpu_identity(
            "vllm",
            local_rank=self.local_rank,
            rank=self.rank,
            tp_rank=get_tensor_model_parallel_rank(),
            dp_rank=self.parallel_config.data_parallel_rank,
            device_ids=self.parallel_config.assigned_physical_gpu_ids,
            group_devices=sorted({str(get_world_group().device), str(get_tp_group().device)}),
            engine_world_size=self.parallel_config.world_size,
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        # self.device identifies both the accelerator type and index (unlike a bare int index).
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            # Preserve the explicit CUDA stream on the NCCL path; the gloo fallback accepts
            # stream=None and ignores it. Both broadcast into `weight` in place.
            stream = torch.cuda.current_stream() if self._sync_backend == "nccl" else None
            self._model_update_group.broadcast(weight, src=0, stream=stream)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
