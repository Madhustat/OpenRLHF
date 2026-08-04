# [Roadmap] Intel XPU enablement for OpenRLHF — device-agnostic core, weight-sync, and native XCCL

> Upstream issue draft (v1). v1 = gloo **is** upstreamed as PR-3 (the working default / fallback);
> only the native XCCL PR-4 is blocked. See v2 for the alternative where gloo is not upstreamed.

## Feature request

First-class Intel XPU (GPU) support for OpenRLHF's Ray + vLLM + DeepSpeed training path.

## Motivation

OpenRLHF has mature NVIDIA/CUDA support, but running on Intel XPU currently requires
vendor-specific patches. This roadmap tracks the work to make the core device-agnostic and to add
weight-sync backends for XPU, closing the parity gap with CUDA. The work is deliberately split
into **independent, individually-mergeable PRs**. **XPU runs today** on the first three PRs;
**only the final PR (native XCCL) is blocked** by an upstream stack issue.

## How this is staged (independence between workstreams)

- **PR-1 and PR-2 are fully independent** of each other and of everything below — pure
  portability, no behavioral change, provable no-op on CUDA. Disjoint files; either can merge
  first.
- **PR-3 (gloo weight-sync)** is independent of PR-1/PR-2 and works on XPU **today**. It is the
  auto-selected default and the correct fallback on any platform without a native GPU-to-GPU
  communicator. **No upstream dependency.**
- **PR-4 (native XCCL XPU→XPU)** is the **only** blocked workstream (an Intel XPU software-stack
  version conflict, see below). Built **opt-in** so it cannot destabilize the merged defaults;
  becomes default-on only once a matched stack ships.

The rule followed throughout: **one file → one PR.** A device-API change that lives inside a
feature file travels with that feature's PR, so no file is split across PRs.

## PR-1 — Device-agnostic core (`torch.accelerator`)   **P0 · ready**

Independent, no-op on CUDA. Replaces hard-coded `torch.cuda.*` with `torch.accelerator.*`.

- [ ] Device-API migration in the files whose *only* change is this
      (`torch.cuda.{current_device,empty_cache,synchronize,set_device}` → `torch.accelerator.*`).
      Verified on XPU (GRPO + RM end-to-end) and equivalent on CUDA.

## PR-2 — Optional flash-attn / ring-attention imports   **P1 · ready (independent)**

- [ ] Make ring-attention / flash-attn imports optional so XPU (which has no flash-attn wheel) can
      run. Independent of PR-1; no behavioral change where flash-attn is present.

## PR-3 — Weight synchronization: gloo backend (trainer → vLLM)   **P0 · ready (independent)**

- [ ] **gloo** CPU-staged broadcast for platforms without a native GPU-to-GPU communicator (e.g.
      XPU). **This is the working XPU path today** — auto-selected default, **no upstream
      dependency**. Verified: GRPO training runs to completion (EXIT 0), weight-sync broadcasts
      correctly each step, and the weights genuinely transfer to the vLLM engine (rewards and
      gradients move across steps). Baseline `timing/broadcast ≈ 1 s/step`.

  On **single-XPU (colocate)**, the trainer and vLLM engine are two processes on one physical
  device, and gloo is currently the only mechanism for that transfer. A device-native path there
  (the XPU analog of CUDA-IPC) is being added **upstream in PyTorch** — the allocator-level IPC
  handle is merged to `main` ([pytorch#188789](https://github.com/pytorch/pytorch/pull/188789)) and
  the end-to-end storage-sharing PR is open
  ([pytorch#191725](https://github.com/pytorch/pytorch/pull/191725); tracking
  [torch-xpu-ops#4000](https://github.com/intel/torch-xpu-ops/issues/4000)). When it ships in a
  released torch, OpenRLHF's existing auto-select will pick it up — **no additional OpenRLHF PR is
  required**; gloo simply remains the correct fallback until then.

## PR-4 — Weight synchronization: native XCCL direct XPU→XPU   **P2 · BLOCKED, opt-in**

- [ ] Native **XCCL** direct XPU→XPU broadcast (no CPU hop) for **multi-XPU**. The collective is
      correct in isolation and the unit suite passes, but full end-to-end use is blocked by a
      two-sided version conflict in the Intel XPU stack (below). Ships opt-in
      (`--vllm.sync_backend xccl`); auto-select stays gloo until a matched stack ships.

## Upstream blocker for PR-4 (external — Intel XPU stack version conflict)

Native XCCL weight-sync needs two subsystems live in the same environment: the **XCCL collective**
(weight transfer) and the **vLLM engine** (rollout). On Intel Battlemage (verified on a 2-XPU box)
these are stable on **mutually exclusive** torch versions:

| torch | oneCCL | XCCL collective | vLLM engine init | Net |
|---|---|---|---|---|
| 2.11.0+xpu | 2021.17 | ❌ hangs | ✅ (per Intel LLM Scaler vLLM) | ✗ |
| 2.12.0+xpu | 2021.17 | ❌ segfault ([torch-xpu-ops#4238](https://github.com/intel/torch-xpu-ops/issues/4238)) | ✅ works | ✗ |
| 2.13.0+xpu | 2022.0 | ✅ isolated broadcast verified | ❌ SYCL crash at init + warm-up all_reduce → `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` | ✗ |

Key facts:

- **torch 2.13 fixes the multi-XPU XCCL collective** (`ProcessGroupXCCL` broadcast) that segfaults
  on 2.12 — but requires CCL env vars (`CCL_ZE_IPC_EXCHANGE=sockets`, `FI_PROVIDER=shm`,
  `CCL_ATL_TRANSPORT=ofi`, `CCL_ATL_SHM=1`, `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`) on this hardware.
  (Intel's "no env vars" verification in #4238 was on different silicon.)
- **On torch 2.13 the vLLM-XPU engine no longer initializes:** hard SYCL crash in
  `ProgramManager::addImage`, and vLLM's startup oneCCL warm-up `all_reduce`
  (`xpu_worker.py init_device`) fails with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` under oneCCL
  2022.
- **oneCCL is ABI-pinned to torch:** torch 2.13's XCCL links `libccl.so.2` (oneCCL 2022);
  downgrading to oneCCL 2021.17 makes `import torch` itself segfault. A pip-level pin swap is
  therefore insufficient.
- Related open oneCCL issues with env-var workarounds:
  [uxlfoundation/oneCCL#212](https://github.com/uxlfoundation/oneCCL/issues/212) (stale L0 IPC
  handle → hang), [#213](https://github.com/uxlfoundation/oneCCL/issues/213) (sockets coerced to
  drmfd in containers).
- **Hardware ceiling:** across separate PCIe root ports there is no GPU P2P
  ([intel/compute-runtime#935](https://github.com/intel/compute-runtime/issues/935),
  [#942](https://github.com/intel/compute-runtime/issues/942)), so even a working XCCL path is
  driver-host-staged — "no CPU hop" requires a shared PCIe switch.

**Unblock plan (PR-4 only):** a pip-level `torch>=2.13` pin is **not** sufficient (it trades the
XCCL blocker for a vLLM blocker). Unblocking requires **one matched, pre-validated stack**: either
Intel's [LLM Scaler vLLM](https://github.com/intel/llm-scaler/tree/main/vllm) pin set (e.g. torch
2.11 + oneCCL 2021.17 + oneAPI 2025.3 + vLLM 0.22) with DeepSpeed/OpenRLHF added, or an upstream
vLLM-XPU release built against torch ≥ 2.13. Once such a stack runs both subsystems, re-run
end-to-end XCCL, flip auto-select to XCCL on multi-XPU, and measure XCCL-vs-gloo speed.

## Status summary

- ✅ **XPU runs today** on PR-1 + PR-2 + PR-3 (gloo weight-sync) — no upstream dependency;
  end-to-end verified (GRPO runs to completion, EXIT 0, weights transfer).
- ⏸️ **Only PR-4** (native XCCL, multi-XPU) is blocked — by the torch ↔ oneCCL ↔ vLLM version
  conflict above, **not** by OpenRLHF code.
- No user-facing regression on CUDA at any stage.

Happy to open PR-1…PR-3 now and share the full multi-XPU validation report; PR-4 follows once a
matched stack is available.
