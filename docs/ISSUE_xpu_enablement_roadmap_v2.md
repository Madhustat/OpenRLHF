# [Roadmap] Intel XPU enablement for OpenRLHF — device-agnostic core and native weight-sync

> Upstream issue draft (v2). v2 = gloo is **not** upstreamed as its own PR; it remains an interim
> local fallback while OpenRLHF waits for the native device-to-device weight-sync paths to land
> upstream, then submits PR-3 directly.

## Feature request

First-class Intel XPU (GPU) support for OpenRLHF's Ray + vLLM + DeepSpeed training path.

## Motivation

OpenRLHF has mature NVIDIA/CUDA support, but running on Intel XPU currently requires
vendor-specific patches. This roadmap tracks the work to make the core device-agnostic and to add
a **native** trainer→vLLM weight-sync path on XPU, closing the parity gap with CUDA. The work is
split into **independent, individually-mergeable PRs**. PR-1 and PR-2 are ready now; the
weight-sync PR waits on upstream PyTorch/Intel fixes (below) so that OpenRLHF ships a **native**
device-to-device path rather than a CPU-staged interim.

## How this is staged (independence between workstreams)

- **PR-1 and PR-2 are fully independent** of each other and of everything below — pure
  portability, no behavioral change, provable no-op on CUDA. Disjoint files; either can merge
  first.
- **PR-3 (native XPU weight-sync)** is the only workstream with an **external dependency** — it
  waits on upstream fixes for both the multi-XPU and single-XPU device-native paths (below).
  Built **opt-in** when submitted, so it cannot destabilize the merged defaults.

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

## PR-3 — Native XPU weight synchronization (trainer → vLLM)   **P2 · waiting on upstream**

Goal: a **native** device-to-device weight transfer on XPU (no CPU staging) — matching how
NVIDIA does it (NCCL for multi-GPU, CUDA-IPC for single-GPU colocate). Both XPU equivalents are
being fixed/added upstream, so this PR is opened once they land in a released stack.

- [ ] **Multi-XPU → XCCL** direct XPU→XPU broadcast. Implemented and unit-tested; blocked by a
      torch ↔ oneCCL ↔ vLLM version conflict (below). Unblocks with `torch ≥ 2.13` on a matched
      stack.
- [ ] **Single-XPU (colocate) → tensor-IPC.** Two processes on one physical device need same-device
      tensor IPC (the XPU analog of CUDA-IPC), which is landing upstream in PyTorch — allocator-level
      handle merged to `main` ([pytorch#188789](https://github.com/pytorch/pytorch/pull/188789)),
      end-to-end storage-sharing PR open
      ([pytorch#191725](https://github.com/pytorch/pytorch/pull/191725); tracking
      [torch-xpu-ops#4000](https://github.com/intel/torch-xpu-ops/issues/4000)). Unblocks when it
      ships in a released torch.

> **Note:** XPU already **runs today** using a gloo CPU-staged broadcast as an interim fallback
> (verified: GRPO runs to completion, EXIT 0, weights transfer, `timing/broadcast ≈ 1 s/step`).
> We are intentionally **not** upstreaming the CPU-staged path — the plan is to wait for the native
> device-to-device fixes above and submit PR-3 directly, so OpenRLHF ships the fast path rather than
> a stopgap.

## Upstream dependency for PR-3 (external — Intel XPU stack)

**Multi-XPU (XCCL) — version conflict.** The XCCL collective and the vLLM engine are stable on
mutually exclusive torch versions on the current Intel XPU stack (verified on a 2-XPU box):

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

**Single-XPU (tensor-IPC) — feature landing upstream.** The same-device cross-process IPC
mechanism (allocator handle + storage sharing) is in progress on PyTorch `main`
([#188789](https://github.com/pytorch/pytorch/pull/188789) merged,
[#191725](https://github.com/pytorch/pytorch/pull/191725) open); it depends on oneAPI 2026.0 and an
experimental SYCL IPC extension, and is not yet in a released torch.

**Unblock plan:** (1) multi-XPU — a matched, pre-validated stack where XCCL and vLLM coexist
(either Intel's [LLM Scaler vLLM](https://github.com/intel/llm-scaler/tree/main/vllm) pin set with
DeepSpeed/OpenRLHF added, or an upstream vLLM-XPU build against `torch ≥ 2.13`); (2) single-XPU —
a released torch containing the XPU tensor-IPC support above. Once available, submit PR-3, enable
auto-select on XPU, and measure native-vs-gloo speed.

## Status summary

- ✅ **XPU runs today** on PR-1 + PR-2 (device-agnostic core + optional imports), using a gloo
  CPU-staged weight-sync as an interim fallback (end-to-end verified: GRPO runs to completion,
  EXIT 0, weights transfer).
- ⏸️ **PR-3 (native weight-sync)** waits on the upstream fixes above — multi-XPU XCCL (version
  conflict) and single-XPU tensor-IPC (feature landing on PyTorch `main`). Not blocked by OpenRLHF
  code.
- No user-facing regression on CUDA at any stage.

Happy to open PR-1 and PR-2 now, and PR-3 once a matched stack / released torch is available; I can
share the full multi-XPU validation report meanwhile.
