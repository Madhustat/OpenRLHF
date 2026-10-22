import os
import socket
from abc import ABC
from dataclasses import fields
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, PolicyLoss, aggregate_loss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from openrlhf.utils.distributed_util import (
    ProcessWatchdog,
    XcclUnavailableError,
    check_xccl_participants,
    resolve_vllm_sync_backend,
    stateless_init_process_group,
    torch_dist_barrier_and_accelerator_sync,
    xccl_preflight,
    xpu_identity,
)
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.loss_utils import get_loss_batch_info, iter_grad_accum_global_norm
from openrlhf.utils.vlm_utils import merge_mm_train_inputs

from ..ppo_utils import NaiveReplayBuffer

logger = init_logger(__name__)

# Failed XCCL weight-sync groups, kept referenced on purpose: destroying a ProcessGroupXCCL blocks until
# every rank destroys it, and the vLLM ranks keep theirs. The launcher stops this process instead.
_FAILED_XCCL_GROUPS = []

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


class ActorPPOTrainer(ABC):
    _sync_watchdog = None  # watchdog of the XCCL weight sync in progress

    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.train.max_epochs

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.actor.eps_clip_low_high[0],
            clip_eps_high=self.args.actor.eps_clip_low_high[1],
            dual_clip=self.args.actor.dual_clip,
            token_level_loss=self.args.actor.loss_agg_mode == "token-mean",
            policy_loss_type=self.args.actor.policy_loss_type,
            is_correction_level=self.args.algo.advantage.is_correction_level,
            is_correction_mode=self.args.algo.advantage.is_correction_mode,
            is_correction_gating=self.args.algo.advantage.is_correction_gating,
            is_correction_threshold=self.args.algo.advantage.is_correction_threshold,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            self.args.ds.packing_samples,
            self.args.train.dynamic_batch_enable,
        )

        # Init torch group for weights sync. Resolve the backend once (auto-detect when unset,
        # strict on explicit) so downstream decisions - especially CUDA IPC - use the effective
        # backend, not the raw config value.
        configured_backend = getattr(self.strategy.args.vllm, "sync_backend", None)
        colocated = self.args.train.colocate_all and not self.args.train.async_enable
        self.vllm_sync_backend = resolve_vllm_sync_backend(configured_backend, colocated=colocated)
        self.use_cuda_ipc = self.vllm_sync_backend == "nccl" and colocated
        self._xccl_summary = f"DeepSpeed rank {torch.distributed.get_rank()}"

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            self._init_vllm_sync_group(self.vllm_sync_backend)

        torch_dist_barrier_and_accelerator_sync()

    def _init_vllm_sync_group(self, backend: str):
        """Create a torch process group between DeepSpeed rank 0 and all vLLM engine ranks.

        Layout example (3 engines, TP=4):
            [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
            |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |

        ZeRO-1/2: broadcast params from rank 0 to all engines.
        ZeRO-3:   allgather to rank 0 first, then broadcast.
        """
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        vllm_num_engines = self.strategy.args.vllm.num_engines
        vllm_tensor_parallel_size = self.strategy.args.vllm.tensor_parallel_size
        world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

        use_ray = getattr(self.strategy.args.vllm, "sync_with_ray", False)
        xccl_checks = backend == "xccl" and not use_ray
        if xccl_checks:
            self._check_xpu_placement(vllm_tensor_parallel_size)
        elif backend == "gloo" and getattr(torch.version, "xpu", None):
            # gloo on XPU is pinned the same way (device index, all XPUs visible): check the placement too
            self._check_xpu_placement(vllm_tensor_parallel_size, xccl=False)
        group_name = "openrlhf"
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * vllm_tensor_parallel_size + 1,
                world_size,
                group_name,
                backend=backend,
                use_ray=use_ray,
            )
            for i, engine in enumerate(self.vllm_engines)
        ]
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            accelerator = torch.accelerator.current_accelerator()
            device = torch.device(accelerator.type, torch.accelerator.current_device_index())
            self._model_update_group = stateless_init_process_group(
                master_address, master_port, 0, world_size, device, backend=backend
            )

        if not xccl_checks:
            ray.get(refs)
            return
        # Real weights go over XCCL only after the test broadcasts arrive bit-exact on every rank.
        trainer = self._xccl_participants[0]
        identity = f"trainer rank 0, {device}, {trainer['uuid']}"
        # If the watchdog ends this process, the launcher reads this marker to fall back to gloo.
        attempt = getattr(self.strategy.args.vllm, "xccl_attempt", None)
        marker = dict(
            attempt=attempt, stage="setup", actor="trainer", rank=0, uuid=trainer["uuid"], pci=trainer["pci"]
        )
        results = {"trainer rank 0": xccl_preflight(self._model_update_group, 0, device, identity, marker=marker)}
        try:
            per_engine = ray.get(refs, timeout=180)
        except ray.exceptions.GetTimeoutError:
            per_engine = None
        except torch.OutOfMemoryError:
            raise  # not an XCCL failure: the job stops with the original error
        except Exception as e:  # a vLLM rank died or raised during its preflight: the group is unusable
            _FAILED_XCCL_GROUPS.append(self._model_update_group)
            raise XcclUnavailableError(f"a vLLM rank failed during the XCCL preflight broadcasts: {e}") from e
        if per_engine is not None:
            for i, per_worker in enumerate(per_engine):
                results.update({f"vLLM engine {i} worker {j}": r for j, r in enumerate(per_worker)})
        failed = [f for checks in results.values() for f in checks if not f["ok"]]
        if per_engine is None or failed:
            _FAILED_XCCL_GROUPS.append(self._model_update_group)
            if per_engine is None:
                raise XcclUnavailableError("vLLM ranks did not finish the XCCL preflight broadcasts within 180 s")
            raise XcclUnavailableError(f"XCCL preflight broadcasts returned wrong data: {failed}")
        logger.info(f"XCCL weight-sync preflight passed on {len(results)} ranks (4 KiB, 16 MiB, 128 MiB bit-exact)")

    def _check_xpu_placement(self, vllm_tensor_parallel_size, xccl=True):
        """Before any XCCL traffic: every weight-sync rank must run on its Ray-assigned XPU, keep its
        memory there and see every communicator XPU. Placement is compared by UUID / PCI address."""
        trainer = xpu_identity(
            "trainer",
            weight_sync_rank=0,
            ray_gpu_ids=[int(g) for g in ray.get_gpu_ids()],
            actor_id=ray.get_runtime_context().get_actor_id(),
        )
        trainer["assigned_xpu"] = trainer["ray_gpu_ids"][0]
        participants = [trainer]
        for i, engine in enumerate(ray.get([e.xpu_identity.remote() for e in self.vllm_engines])):
            for worker in engine["workers"]:
                # device_ids lists the worker's XPUs by vLLM local position (its TP rank on one node);
                # without it (gloo-style masking) the actor's own Ray assignment applies.
                device_ids = worker["device_ids"] or engine["ray_gpu_ids"]
                participants.append(
                    {
                        **worker,
                        "actor_id": engine["actor_id"],
                        "ray_gpu_ids": engine["ray_gpu_ids"],
                        "weight_sync_rank": i * vllm_tensor_parallel_size + 1 + worker["rank"],
                        "assigned_xpu": device_ids[worker["tp_rank"]],
                    }
                )
        for p in participants:
            logger.info(
                f"XCCL weight-sync rank {p['weight_sync_rank']}: {p['role']} pid {p['pid']} actor {p['actor_id']} "
                f"Ray XPU {p['ray_gpu_ids']} -> {p['name']} {p['uuid']} ({p['pci']}), visible {p['visible_pci']}, "
                f"ZE_AFFINITY_MASK={p['ZE_AFFINITY_MASK']} ONEAPI_DEVICE_SELECTOR={p['ONEAPI_DEVICE_SELECTOR']}"
                + (
                    f", vLLM local/TP/DP rank {p['local_rank']}/{p['tp_rank']}/{p['dp_rank']}, device_ids {p['device_ids']}, "
                    f"process groups on {p.get('group_devices')}"
                    if p["role"] == "vllm"
                    else ""
                )
            )
        # Ray's XPU IDs are Level Zero enumeration indices; the trainer sees every XPU (checked).
        check_xccl_participants(
            participants,
            trainer["visible_uuids"],
            colocated=self.args.train.colocate_all and not self.args.train.async_enable,
            xccl=xccl,
            reference_pci=trainer["visible_pci"],
        )
        self._xccl_participants = participants
        self._xccl_summary = "; ".join(
            f"rank {p['weight_sync_rank']} {p['role']} {p['uuid']} ({p['pci']})" for p in participants
        )

    def ppo_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = (
            self.strategy.ring_attn_group is None
            and self.args.ds.tensor_parallel_size <= 1
            and not self.args.train.dynamic_batch_enable
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=should_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            # Pair each experience with its loss normalization. Dynamic batching computes it
            # from the replay buffer inside training_step (loss_batch_info=None); otherwise we
            # normalize by the optimizer-step window's global action-token count.
            if self.args.train.dynamic_batch_enable:
                micro_batches = ((exp, None) for exp in pbar)
            else:
                micro_batches = iter_grad_accum_global_norm(
                    pbar, self.strategy, self.strategy.accumulated_gradient, lambda e: e.action_mask
                )
            for step, (experience, loss_batch_info) in enumerate(micro_batches):

                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step, loss_batch_info)

                metrics = status["metrics"]
                weights = status["weights"]
                n_tokens = status["num_action_tokens"]
                n_samples = status["num_samples"]

                reduced_status = {"_num_action_tokens": n_tokens, "_num_samples": n_samples}
                last_metrics = {}
                for k, value in metrics.items():
                    weight = weights[k]
                    if weight is None:
                        last_metrics[k] = value
                        continue
                    scale = n_tokens if weight == "token" else n_samples
                    reduced_status[k] = (
                        value.float().mean().item() * scale if isinstance(value, torch.Tensor) else value * scale
                    )

                reduced_status = self.strategy.all_reduce(reduced_status)

                n_tokens = reduced_status.pop("_num_action_tokens")
                n_samples = reduced_status.pop("_num_samples")
                merged_status = {}
                for k, value in reduced_status.items():
                    denom = n_tokens if weights[k] == "token" else n_samples
                    merged_status[k] = value / denom

                merged_status.update(last_metrics)
                merged_status["_num_samples"] = n_samples
                merged_status["_num_action_tokens"] = n_tokens
                merged_status["_weights"] = weights
                actor_lr = merged_status.get("actor_lr", 0)

                short_status = {
                    "act_loss": merged_status["policy_loss"],
                    "reward": merged_status.get("reward", 0),
                    "return": merged_status.get("return", 0),
                    "gen_len": merged_status.get("response_length", 0),
                    "tot_len": merged_status.get("total_length", 0),
                    "kl": merged_status.get("kl", 0),
                    "act_lr": actor_lr,
                    "grad_norm": merged_status.get("actor_grad_norm", 0),
                }
                if "entropy_loss" in merged_status:
                    short_status["ent_loss"] = merged_status["entropy_loss"]

                status_list.append(merged_status)
                pbar.set_postfix(short_status)

        if status_list:
            total_tokens = sum(s["_num_action_tokens"] for s in status_list)
            total_samples = sum(s["_num_samples"] for s in status_list)
            status_mean = {}
            for k in set().union(*(s.keys() for s in status_list)):
                if k in ("_num_samples", "_num_action_tokens", "_weights"):
                    continue
                if k in ("actor_grad_norm", "actor_lr"):
                    vals = [s[k] for s in status_list if k in s]
                    status_mean[k] = vals[-1] if vals else 0.0
                elif status_list[0].get("_weights", {}).get(k) == "token":
                    status_mean[k] = sum(s.get(k, 0) * s["_num_action_tokens"] for s in status_list) / total_tokens
                else:
                    status_mean[k] = sum(s.get(k, 0) * s["_num_samples"] for s in status_list) / total_samples
        return status_mean

    def training_step(
        self, experience: Experience, kl_ctl: float, step: int, loss_batch_info: Optional[Dict] = None
    ) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs
        # loss_batch_info is precomputed per optimizer-step window on the non-dynamic path
        # (global token-mean over the step); the dynamic path computes it from the replay buffer.
        if loss_batch_info is None:
            loss_batch_info = get_loss_batch_info(
                self.strategy,
                action_mask,
                replay_buffer=self.replay_buffer,
                step=step,
                dynamic_batch=self.args.train.dynamic_batch_enable,
            )

        # VLM: merge pre-processed multimodal inputs for training forward
        mm_inputs = {}
        if (
            hasattr(experience, "mm_train_inputs")
            and experience.mm_train_inputs
            and getattr(self.actor, "is_vlm", False)
        ):
            mm_inputs = merge_mm_train_inputs(experience.mm_train_inputs, sequences.device)

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.actor.entropy_coef is not None,
            **mm_inputs,
        )

        # loss function
        actor_loss, clip_ratio, ppo_kl, vllm_kl, is_filter_ratio = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
            **loss_batch_info,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()
            experience.info["is_filter_ratio"] = is_filter_ratio.detach()

        if self.args.algo.kl.use_loss:
            if self.args.algo.kl.init_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    log_probs_old=old_action_log_probs,
                    kl_estimator=self.args.algo.kl.estimator,
                    unbiased_gradient=self.args.algo.kl.unbiased_gradient,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs)
                logprobs_diff = torch.zeros_like(action_log_probs)
            kl_loss = aggregate_loss(kl, experience.action_mask, **loss_batch_info)
            logprobs_diff = masked_mean(logprobs_diff, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
            experience.info["logprobs_diff"] = logprobs_diff.detach()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * kl_ctl
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss * self.args.actor.aux_loss_coef
            if self.args.train.dynamic_batch_enable:
                aux_loss = aux_loss * self.replay_buffer.dynamic_sample_loss_scale[step]
            loss += aux_loss
        # entropy loss
        if self.args.actor.entropy_coef is not None:
            entropy_loss = aggregate_loss(
                output.entropy[:, -experience.action_mask.shape[1] :],
                experience.action_mask,
                **loss_batch_info,
            )
            if self.args.actor.entropy_coef != 0:
                loss -= entropy_loss * self.args.actor.entropy_coef

        self.strategy.backward(loss, self.actor, self.actor_optim)
        if self.args.train.dynamic_batch_enable:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.train.dynamic_batch_enable:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # Per-token losses (0-D tensors, shape carries weighting info for ppo_train)
        metrics = {"policy_loss": actor_loss.detach()}
        weights = {"policy_loss": "token"}
        if self.args.actor.entropy_coef is not None:
            metrics["entropy_loss"] = entropy_loss.detach()
            weights["entropy_loss"] = "token"

        # Non-reducible meta
        metrics["actor_lr"] = self.actor_scheduler.get_last_lr()[0]
        weights["actor_lr"] = None
        is_optimizer_step = not self.args.train.dynamic_batch_enable or self.replay_buffer.dynamic_optimizer_step[step]
        if is_optimizer_step:
            metrics["actor_grad_norm"] = self.strategy.get_grad_norm(self.actor)
            weights["actor_grad_norm"] = None

        # Merge all loggable tensors.
        # `info` keeps algorithm metrics; episode tensor fields are added explicitly below.
        for k, v in experience.info.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v
                weights[k] = "token" if v.dim() == 0 else "sample"
            elif isinstance(v, list):
                metrics[k] = torch.tensor(v, dtype=torch.float)
                weights[k] = "sample"

        for f in fields(Experience):
            if f.name in {"rewards", "scores"} or not Experience.is_episode_tensor_field(f.name):
                continue
            value = getattr(experience, f.name)
            if isinstance(value, torch.Tensor) and f.name not in metrics:
                metrics[f.name] = value
                weights[f.name] = "sample"

        return {
            "metrics": metrics,
            "weights": weights,
            "num_samples": float(experience.action_mask.shape[0]),
            "num_action_tokens": float(experience.action_mask.sum().item()),
        }

    def broadcast_to_vllm(self):
        if self.vllm_sync_backend != "xccl":
            return self._broadcast_to_vllm()
        # A hung XCCL weight sync would block forever (ProcessGroupXCCL reports completion early and
        # its timeout does not fire): bound each tensor's transfer, and stop rather than train on stale weights.
        timeout = getattr(self.strategy.args.vllm, "sync_timeout", 600)
        with ProcessWatchdog(timeout, f"XCCL weight sync to vLLM [{self._xccl_summary}]") as self._sync_watchdog:
            self._broadcast_to_vllm()
        self._sync_watchdog = None

    def _broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args.vllm, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.accelerator.empty_cache()
        model = self.actor.model.module
        count = 0

        def _broadcast_param(param, count, num_params):
            if self._sync_watchdog:
                self._sync_watchdog.restart(f"vLLM update and broadcast of {name}")
            use_ray = getattr(self.strategy.args.vllm, "sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.ds.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    # Preserve the explicit CUDA stream on the NCCL path; the gloo fallback
                    # accepts stream=None and ignores it. Both communicators broadcast into
                    # param.data in place and return the same tensor.
                    stream = torch.cuda.current_stream() if self.vllm_sync_backend == "nccl" else None
                    self._model_update_group.broadcast(param.data, src=0, stream=stream)
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.ds.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_accelerator_sync()

        def _gather_params_ctx(param):
            """Context manager that gathers sharded/TP-split parameters for weight sync."""
            if self.strategy.args.ds.tensor_parallel_size > 1:
                return deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True)
            return deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.ds.zero_stage == 3)

        sync_fn = _handle_cuda_ipc if self.use_cuda_ipc else _broadcast_param

        # VLM: only sync trainable (language model) params — vision encoder is frozen.
        params_to_sync = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad or not getattr(self.actor, "is_vlm", False)
        ]
        num_params = len(params_to_sync)

        for name, param in params_to_sync:
            count += 1  # empty_cache at last param
            with _gather_params_ctx(param):
                sync_fn(param, count, num_params)

        if self._sync_watchdog:
            self._sync_watchdog.restart("final synchronize and barrier")
        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.accelerator.empty_cache()
        torch_dist_barrier_and_accelerator_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.ckpt.save_hf
        self.disable_ds_ckpt = args.ckpt.disable_ds
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        # Skip for vLLM >= 0.16 where NCCL_CUMEM_ENABLE=0 causes ncclCommInitRank to fail
        # with "unhandled cuda error" under NCCL 2.27+.
        # Only relevant when vLLM engines are actually in use - vllm may not be
        # installed at all when running with --vllm.num_engines 0.
        colocated = args.train.colocate_all and not args.train.async_enable
        if vllm_engines and resolve_vllm_sync_backend(getattr(args.vllm, "sync_backend", None), colocated) == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.ds.attn_implementation,
            experts_implementation=strategy.args.ds.experts_implementation,
            param_dtype=strategy.args.ds.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.ds.load_in_4bit,
            lora_rank=strategy.args.ds.lora.rank,
            lora_alpha=strategy.args.ds.lora.alpha,
            target_modules=strategy.args.ds.lora.target_modules,
            lora_dropout=strategy.args.ds.lora.dropout,
            ds_config=strategy.get_ds_train_config(),
            packing_samples=strategy.args.ds.packing_samples,
            temperature=strategy.args.rollout.temperature,
            use_liger_kernel=strategy.args.ds.use_liger_kernel,
            freeze_visual_encoder=getattr(strategy.args.actor, "freeze_visual_encoder", False),
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
        )

        if args.train.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.ds.attn_implementation,
                experts_implementation=strategy.args.ds.experts_implementation,
                param_dtype=strategy.args.ds.param_dtype,  # default: bf16
                load_in_4bit=strategy.args.ds.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.ds.packing_samples,
            )
        else:
            ema_model = None

        if args.actor.gradient_checkpointing_enable:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )

        actor_cfg = dict(
            optim=args.actor.optim,
            muon=vars(args.actor.muon),
            adam=vars(args.actor.adam),
            lr_scheduler=args.actor.lr_scheduler,
            lr_warmup_ratio=args.actor.lr_warmup_ratio,
            min_lr_ratio=args.actor.min_lr_ratio,
            max_norm=args.actor.max_norm,
            scheduler_steps=max_steps,
        )
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare((actor, actor_cfg))

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt.path, "_actor")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states = states

        # initial offload
        if strategy.args.ds.enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.actor.eps_clip,
            ema_beta=args.train.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.train.enable_ema else self.actor,
            self.tokenizer,
            args.ckpt.output_dir,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        mm_train_inputs_list=None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()

        # VLM: merge pre-processed multimodal inputs from all samples in batch
        mm_inputs = {}
        if mm_train_inputs_list and getattr(self.actor, "is_vlm", False):
            mm_inputs = merge_mm_train_inputs(mm_train_inputs_list, device)

        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                **mm_inputs,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        self.trainer.broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
        args = self.strategy.args
        client_states = client_states or {}
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt.path, "_actor"),
                tag,
                args.ckpt.max_num,
                args.ckpt.max_mem,
                client_states,
                metric_value=metric_value,
                metric_key=metric_key,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt.path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.train.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_accelerator_sync()
