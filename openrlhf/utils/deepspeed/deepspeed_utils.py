import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from packaging import version


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    param_dtype="bf16",
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    overlap_comm=False,
    use_ds_universal_ckpt=False,
    deepcompile=False,
    tensor_parallel_size=1,
    optim_config=None,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
    elif stage == 3:
        # DeepSpeed's own ZeRO config resolves overlap_comm=None to
        # `self.stage == ZeroStageEnum.weights` (i.e. True for stage 3) whenever the key is
        # absent -- so omitting it here to mean "False" silently becomes "True" for stage 3,
        # the opposite of this function's own overlap_comm=False default. Confirmed root cause
        # (2026-09) of a native SIGSEGV in oneCCL's background progress thread
        # (ccl_worker_func -> ... -> urEventGetInfo -> libze_intel_gpu.so.1) on 2x Intel
        # Battlemage B70 (PCIe, no XeLink) under the ATL/OFI transport -- a minimal DeepSpeed
        # ZeRO-3 reproducer crashes 3/3 with this key absent and passes 3/3 (300 steps) with it
        # explicitly set False, all other variables held constant. Must be written explicitly.
        zero_opt_dict["overlap_comm"] = False
    if stage == 3:
        zero_opt_dict["reduce_scatter"] = True

    ds_config = {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": param_dtype == "bf16",
        },
        "fp16": {
            "enabled": param_dtype == "fp16",
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype},
        "checkpoint": {
            "load_universal": use_ds_universal_ckpt,
        },
        "compile": {
            "deepcompile": deepcompile,
        },
        "tensor_parallel": {
            "autotp_size": tensor_parallel_size,
        },
    }

    # Optimizer — always config-based, DeepSpeed creates it from this config.
    # DS auto-selects FusedAdam / DeepSpeedCPUAdam based on offload setting.
    if optim_config is not None:
        ds_config["optimizer"] = optim_config

    return ds_config


def get_eval_ds_config(
    offload,
    stage=0,
    param_dtype="bf16",
    deepcompile=False,
    tensor_parallel_size=1,
):
    # At least for 0.16.6, DeepCompile hasn't support pure inference mode
    # https://github.com/deepspeedai/DeepSpeed/pull/7225
    deepcompile = False

    zero_opt_dict = {
        "stage": stage,
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": param_dtype == "bf16",
        },
        "fp16": {
            "enabled": param_dtype == "fp16",
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "compile": {
            "deepcompile": deepcompile,
        },
        "tensor_parallel": {
            "autotp_size": tensor_parallel_size,
        },
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def _optimizer_supports_state_offload(model):
    """DeepSpeedZeroOptimizer_Stage3.offload_states() hard-asserts
    `self.optimizer.__class__ == deepspeed.ops.adam.fused_adam.FusedAdam` before touching either
    OffloadStateTypeEnum.optim_states or .hp_params (deepspeed/runtime/zero/stage3.py, inside
    offload_states()) -- both categories call into FusedAdam-specific helpers
    (offload_adam_states / the fp32 master-weight buffers). The other three categories
    (lp_params, lp_grads, contiguous_grad_buffer) never reference self.optimizer at all; they
    only move DeepSpeed's own ZeRO-3 partition buffers, so they work with any optimizer.
    """
    inner_optimizer = getattr(model.optimizer, "optimizer", None)
    return inner_optimizer is not None and inner_optimizer.__class__ is deepspeed.ops.adam.fused_adam.FusedAdam


def offload_deepspeed_states(model, pin_memory=True, non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3 and version.parse(deepspeed.__version__) <= version.parse("0.17.5"):
        raise NotImplementedError(
            "Only Zero stage 3 is currently supported when using DeepSpeed version 0.17.5 or lower"
        )

    # if zero_stage == 3 and not adam_offload:
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum

    fused_adam_capable = _optimizer_supports_state_offload(model)

    offload_state_types = [
        OffloadStateTypeEnum.contiguous_grad_buffer,
    ]
    if fused_adam_capable:
        offload_state_types += [
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.hp_params,
        ]

    if version.parse(deepspeed.__version__) >= version.parse("0.16.5"):
        # These offload types are fixed in https://github.com/deepspeedai/DeepSpeed/pull/7050
        offload_state_types += [
            OffloadStateTypeEnum.lp_grads,
            # OffloadStateTypeEnum.lp_params,
        ]

    if not fused_adam_capable and not getattr(model, "_openrlhf_partial_sleep_logged", False):
        inner = getattr(model.optimizer, "optimizer", None)
        # Capability-based fallback, not a silent downgrade: this optimizer can't do the
        # FusedAdam-only optim_states/hp_params offload, so those two categories stay resident
        # on-device while everything else (gradients, the contiguous grad buffer, and -- on
        # DeepSpeed >= 0.16.5 -- low-precision grads) is still offloaded. Confirmed root cause
        # (2026-09) for why `--ds.enable_sleep` used to hard-crash with plain torch.optim.AdamW:
        # `stage3.py:3285 AssertionError: Offloading is supported only for DeepSpeed FusedAdam`.
        print(
            f"[deepspeed sleep] partial offload mode: optimizer={type(inner).__name__} does not "
            f"support FusedAdam-only state offload -- retaining optim_states, hp_params on-device; "
            f"offloading {[t.name for t in offload_state_types]}",
            flush=True,
        )
        model._openrlhf_partial_sleep_logged = True

    model.optimizer.offload_states(
        include=offload_state_types,
        device=OffloadDeviceEnum.cpu,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )
    model.empty_partition_cache()
    torch.accelerator.empty_cache()
    torch.distributed.barrier()
    torch.accelerator.synchronize()


def reload_deepspeed_states(model, non_blocking=True):
    zero_stage = model.zero_optimization_stage()  # config['zero_optimization']['stage']
    adam_offload = model.config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"

    # state offloading not required when using Adam optimizer offloading
    if adam_offload:
        return

    if zero_stage != 3 and version.parse(deepspeed.__version__) <= version.parse("0.17.5"):
        raise NotImplementedError(
            "Only Zero stage 3 is currently supported when using DeepSpeed version 0.17.5 or lower"
        )

    # if zero_stage == 3 and not adam_offload:
    import torch

    model.reload_states(non_blocking=non_blocking)
    torch.accelerator.empty_cache()
    torch.distributed.barrier()
    torch.accelerator.synchronize()
