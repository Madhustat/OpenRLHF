# [Roadmap] OpenRLHF on Intel XPU

## Feature request

First-class Intel XPU support for OpenRLHF's Ray + vLLM + DeepSpeed training path.

## Motivation

OpenRLHF has mature NVIDIA/CUDA support, but Intel XPU currently requires downstream patches and a CPU-staged Gloo fallback for trainer-to-vLLM weight synchronization.

The portability changes can be submitted now. Native XPU weight synchronization will follow once the required upstream PyTorch/XPU support is available in a compatible released stack.

## Device-agnostic core **P0**

- [ ] Replace applicable hard-coded `torch.cuda.*` calls with device-agnostic `torch.accelerator.*` APIs.
- [ ] Validate the affected training paths on Intel XPU without changing existing CUDA behavior.

## Optional attention dependencies **P1**

- [ ] Make flash-attn and ring-attention imports optional so OpenRLHF can run on XPU without these CUDA-specific packages.
- [ ] Preserve the current behavior when these dependencies are installed.

## Native trainer-to-vLLM weight synchronization **P2: waiting on upstream**

OpenRLHF runs the trainer and vLLM rollout engines as separate Ray actors, so updated policy weights must be synchronized during online RL training.

XPU currently works through a downstream Gloo fallback:

```text
trainer XPU -> CPU/Gloo -> vLLM XPU
```

This fallback stays downstream and is not proposed as a separate upstream PR. Native device-to-device paths will be submitted once upstream support ships in a compatible released stack:

- [ ] **Multi-XPU: XCCL** direct XPU→XPU broadcast — waiting on a compatible XCCL + vLLM-XPU stack ([torch-xpu-ops#4238](https://github.com/intel/torch-xpu-ops/issues/4238)).
- [ ] **Single-XPU (colocation): tensor-IPC** — waiting on cross-process XPU tensor IPC in PyTorch ([torch-xpu-ops#4000](https://github.com/intel/torch-xpu-ops/issues/4000)).

## Notes

- The device-agnostic core and optional-attention changes are independent of the native      weight-synchronization work and can be reviewed separately.
- No user-facing change on CUDA is intended at any stage.
