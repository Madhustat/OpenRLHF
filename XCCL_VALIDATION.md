# Direct XPU-to-XPU XCCL Weight Synchronization — Validation Guide

This document describes how to enable and validate direct XCCL weight
synchronization between the DeepSpeed actor and the vLLM rollout engine on
Intel XPU hardware, using an isolated Docker environment.

---

## Background

OpenRLHF normally synchronizes actor weights to vLLM via a CPU-staged Gloo
broadcast after each optimizer step. On Intel XPU this works but routes every
parameter tensor through host memory. This document covers the validated path
for replacing Gloo with direct XPU-to-XPU XCCL weight transfer:

```
Actor (XPU 0)  --[XCCL broadcast, no CPU hop]-->  vLLM engine (XPU 1)
```

---

## Environment

### Hardware

- 2× Intel Arc Pro B70 (Battlemage, PCIe)
- XPU 0: PCI `0000:18:00.0` — policy actor + reference model
- XPU 1: PCI `0000:54:00.0` — vLLM rollout engine

### Verify both XPUs are healthy before starting

```bash
python -c "
import torch
for i in range(torch.xpu.device_count()):
    x = torch.ones(1024, device=f'xpu:{i}')
    torch.xpu.synchronize(i)
    y = (x * 2).sum()
    torch.xpu.synchronize(i)
    print(f'XPU {i}: {y.cpu().item()} — PASS')
"
```

Expected output:
```
XPU 0: 2048.0 — PASS
XPU 1: 2048.0 — PASS
```

> Run from an XPU-enabled Python environment. If either device fails, reboot
> the machine and retest before proceeding — GPU driver stuck states can
> mimic hardware failures and typically clear on a clean reboot.

---

## Base Image

```bash
docker pull vllm/vllm-openai-xpu:v0.27.1
```

This is the official vLLM-published Docker image providing:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.13.0+xpu |
| vLLM | 0.27.1 |
| Intel runtime | 2026.0.0 |
| oneCCL | 2022.0.0 |
| Ray | 2.57.0 |

This image is used as the validation base because a pip-installable
`torch 2.13 + vLLM 0.27` XPU pairing is not yet available. The OpenRLHF
code changes identified here apply equally to a native pip installation once
those wheels are released.

---

## Required Patches

Two upstream issues must be patched before the stack works. Both are applied
as thin image layers on top of `vllm/vllm-openai-xpu:v0.27.1`.

### Patch 1 — vLLM issue #52386 / PR #52389

vLLM 0.27.1 performs an unconditional oneCCL `all_reduce` warm-up in
`xpu_worker.py` at world_size=1. This hangs indefinitely on this hardware
because the warm-up is only meaningful for multi-device inference but fires
even when a single vLLM engine is used.

**Fix:** add a `world_size > 1` guard.

```python
# /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py
# Before:
if torch.distributed.is_xccl_available():
    torch.distributed.all_reduce(torch.zeros(1).xpu())

# After (PR #52389):
if (
    self.parallel_config.world_size > 1
    and torch.distributed.is_xccl_available()
):
    torch.distributed.all_reduce(torch.zeros(1).xpu())
```

### Patch 2 — DeepSpeed FusedAdam fallback

The vLLM serving image does not bundle the Intel C++ compiler (`icpx`), so
DeepSpeed cannot JIT-compile its XPU-optimized `fused_adam` extension.
Without a fallback the actor crashes at optimizer initialization.

**Fix:** catch the JIT failure and fall back to `torch.optim.AdamW`.

```python
# /opt/venv/lib/python3.12/site-packages/deepspeed/runtime/engine.py
# Replace the FusedAdam selection block with:
try:
    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(model_parameters, **optimizer_parameters,
                          adam_w_mode=effective_adam_w_mode)
except Exception:
    import torch
    optimizer = torch.optim.AdamW(
        model_parameters,
        **{k: v for k, v in optimizer_parameters.items()
           if k in ("lr", "betas", "eps", "weight_decay")}
    )
```

> This workaround is specific to the Docker image. On a native installation
> with the full oneAPI toolkit, `icpx` is on PATH and `fused_adam` compiles
> normally.

---

## Build the Training Image

Apply both patches as layers on top of the base image:

```bash
# patch_vllm_xpu_worker.py  — apply PR #52389
# patch_ds_adam.py          — apply AdamW fallback
# (see scripts/ in this repo)

docker build -t openrlhf-xccl:v0271 .
```

The `Dockerfile` in this directory applies both patches and symlinks
`ninja` to the system PATH so DeepSpeed's build system can find it in Ray
worker subprocesses.

---

## OpenRLHF Code Changes

One set of changes is required in OpenRLHF itself. These are not
Docker-specific — they apply to any XPU installation.

### `torch.cuda.*` → `torch.accelerator.*`

Three calls in `ppo_actor.py` and `vllm_worker_wrap.py` were not covered by
the original device-agnostic PR. They only surface when the XCCL training
path is exercised (not reached by the Gloo-based test suite).

| File | Line | Before | After |
|---|---|---|---|
| `ppo_actor.py` | 183 | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` |
| `ppo_actor.py` | 640 | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` |
| `ppo_actor.py` | 426, 501, 613, 617 | `torch.cuda.empty_cache()` | `torch.accelerator.empty_cache()` |
| `ppo_actor.py` | 618 | `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` |
| `vllm_worker_wrap.py` | 74 | `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` |
| `distributed_util.py` | 15 | `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` |

These changes are included in the `docker-experiment` branch and will be
submitted as a follow-up to the upstream device-agnostic PR.

---

## Docker Run Configuration

The container requires specific device and security flags for oneCCL IPC to
initialize correctly:

```bash
docker run --rm \
  --entrypoint bash \
  --device=/dev/dri \
  --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  --ipc=host \
  --shm-size=32g \
  -v $(pwd):/repo \
  -v /tmp/hf_cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  openrlhf-xccl:v0271 \
  /repo/run_xccl_e2e.sh xccl
```

| Flag | Why required |
|---|---|
| `--device=/dev/dri` | exposes the GPU device nodes |
| `--mount /dev/dri/by-path:ro` | required for oneCCL `drmfd` IPC exchange — without it oneCCL cannot enumerate device file descriptors |
| `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` | enables oneCCL `pidfd` IPC exchange (Docker default seccomp blocks `pidfd_getfd`) |
| `--ipc=host --shm-size=32g` | shared memory for Ray object store |

> Either the `by-path` mount (drmfd path) or the seccomp flags (pidfd path)
> is sufficient on its own. Both together is the recommended configuration.

---

## GPU Layout

Two physical XPUs are required. The policy actor and reference model are
colocated on XPU 0; the vLLM engine runs on XPU 1.

```
XPU 0  (PCI 18:00.0)  ←  policy actor + reference model  (ZE_AFFINITY_MASK=0)
XPU 1  (PCI 54:00.0)  ←  vLLM engine                     (ZE_AFFINITY_MASK=1)

XCCL broadcast:  XPU 0  ──────────────────────────────►  XPU 1
                 (actor sends updated weights)            (vLLM receives)
```

The device pinning is handled automatically by OpenRLHF's PR1
`_configure_device_env()` which sets `ZE_AFFINITY_MASK` from
`ray.get_gpu_ids()` before any torch.xpu call.

---

## Training Command

```bash
bash run_xccl_e2e.sh xccl   # direct XCCL
bash run_xccl_e2e.sh gloo   # CPU-staged Gloo (control)
```

Key arguments:

```bash
--vllm.sync_backend xccl          # direct XPU-to-XPU, no CPU hop
--train.colocate_actor_ref        # pack policy+ref on XPU 0 (2-GPU fit)
--critic.model_name_or_path ""    # no separate critic (saves 1 GPU)
--algo.advantage.estimator group_norm   # GRPO without critic
--rollout.n_samples_per_prompt 4  # required for group_norm estimator
```

---

## Validated Result

```
vLLM weight-sync backend: xccl (direct XPU-to-XPU)

✨ Global step 1:
    policy_loss:        0.024
    math_accuracy:      0.066
    timing/broadcast:   1.98 s
    timing/ppo_train:   29.9 s
    timing/generation:  41.2 s

TRAIN_EXIT=0
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| B70 has no PCIe P2P | Both cards sit behind separate PCIe root ports — XCCL is host-staged (~1.8× faster than Gloo but not true zero-copy). True device-to-device transfer requires a shared PCIe switch or Xe Link. |
| Triton JIT warmup | vLLM compiles XPU kernels on first run (~2 min). Subsequent runs reuse the cache. |
| No critic on 2 GPUs | A separate critic model requires a third GPU or `colocate_critic_reward`. |
| Gloo remains default | `--vllm.sync_backend` defaults to `gloo` on XPU. Pass `xccl` explicitly. |

---

## Repository Structure

```
docker-experiment/           ← this branch
├── openrlhf/                ← OpenRLHF with XPU/XCCL changes
│   ├── trainer/ray/
│   │   ├── ppo_actor.py     ← torch.accelerator fixes + XCCL wiring
│   │   ├── vllm_worker_wrap.py  ← torch.accelerator fix + XCCL stream
│   │   └── launcher.py      ← ZE_AFFINITY_MASK actor pinning (PR3)
│   └── utils/
│       └── distributed_util.py  ← _XcclBroadcastCommunicator + resolver
├── run_xccl_e2e.sh          ← validated run script
├── Dockerfile               ← builds openrlhf-xccl:v0271
├── patch_vllm_xpu_worker.py ← applies vLLM PR #52389
├── patch_ds_adam.py         ← applies DeepSpeed AdamW fallback
└── XCCL_VALIDATION.md       ← this document
```
