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

OpenRLHF runs the trainer and vLLM rollout engines as separate Ray actors, so updated policy weights must be synchronized during online RL training. On NVIDIA, OpenRLHF supports NCCL-based synchronization and a CUDA-IPC optimization for colocated actors. On XPU, our downstream implementation currently uses a CPU-staged Gloo fallback:

```text
trainer XPU -> CPU/Gloo -> vLLM XPU
```

The Gloo fallback stays downstream and is not proposed as a separate upstream PR. The native device-to-device paths are being fixed/added upstream, and this work will be submitted once they ship in a compatible released stack:

- [ ] **Multi-XPU: XCCL** direct XPU→XPU broadcast. Blocked by a torch ↔ oneCCL ↔ vLLM compatibility issue on the current XPU stack: `torch >= 2.13` is required but not sufficient on its own, because the versions that fix the XCCL collective and the versions that keep the vLLM-XPU engine working do not currently line up in a single released stack. Unblocking needs a matched, pre-validated torch + oneCCL + oneAPI + vLLM combination (see [torch-xpu-ops#4238](https://github.com/intel/torch-xpu-ops/issues/4238)).
- [ ] **Single-XPU (colocate): tensor-IPC** — the XPU analog of CUDA-IPC, landing upstream in PyTorch ([pytorch/pytorch#188789](https://github.com/pytorch/pytorch/pull/188789), [pytorch/pytorch#191725](https://github.com/pytorch/pytorch/pull/191725); tracking [torch-xpu-ops#4000](https://github.com/intel/torch-xpu-ops/issues/4000)).

## Notes

- The device-agnostic core and optional-attention PRs are intended to be independent of each other and of the weight-sync work, so each can be reviewed on its own; the exact file boundaries will be confirmed when the PRs are opened.
- No user-facing change on CUDA is intended at any stage.
