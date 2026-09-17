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

        # Deep-check #2 (full path): read the just-loaded param back from the vLLM
        # model and assert it bit-matches the broadcast value -> proves the
        # actor->broadcast->vLLM-load path is exact (opt-in). Only params whose
        # names map 1:1 (unfused) are comparable; fused/sharded names are skipped.
        import os
        if os.environ.get("OPENRLHF_DEEPCHECK_SYNC", "0") == "1":
            params = getattr(self, "_dc_params", None)
            if params is None:
                params = {pn: p for pn, p in self.model_runner.model.named_parameters()}
                self._dc_params, self._dc_checked, self._dc_exact = params, 0, 0
            p = params.get(name)
            if p is not None and tuple(p.shape) == tuple(weight.shape):
                self._dc_checked += 1
                if torch.equal(p.data.to(weight.dtype), weight):
                    self._dc_exact += 1
                else:
                    print(f"DEEPCHECK-SYNC-MISMATCH name={name}", flush=True)
                if self._dc_checked <= 3 or self._dc_checked % 100 == 0:
                    print(f"DEEPCHECK-SYNC checked={self._dc_checked} exact={self._dc_exact}", flush=True)

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

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
