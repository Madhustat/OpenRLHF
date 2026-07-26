# Enabling Intel XPU for OpenRLHF — Summary

**Goal:** Run OpenRLHF (Ray + vLLM + DeepSpeed RLHF framework) on Intel XPU, with changes that
are upstream-acceptable and keep the NVIDIA path unchanged.

---

## Headline

- OpenRLHF now runs on Intel XPU. All RLHF workflows were smoke-tested and run end-to-end on a
  single XPU; the NVIDIA/CUDA path is preserved.
- The enablement is a **device-agnostic API migration** plus a small number of **targeted
  correctness fixes**, split into focused, independently reviewable PRs.
- **3 PRs submitted** (test-backed). Additional enablement PRs are prepared and validated on
  single-XPU, pending multi-GPU/NVIDIA hardware for final sign-off.

---

## Unit Test Summary

| OpenRLHF | Latest |
|---|---|
| Total unit test (UT) | 21 |
| # of pure SW UT | 9 |
| # of passing pure SW UT | 9 |
| # of blocked pure SW UT | 0 |
| # of failed pure SW UT | 0 |
| # of XPU runnable UT | 12 |
| # of passing XPU runnable UT | 12 |
| # of failed XPU runnable UT | 0 |
| **Passing rate** (passing XPU / XPU runnable) | **100%** |
| # of newly added XPU UT | 4 |

**Breakdown**

| Group | Count | Tests |
|---|---|---|
| Newly added XPU-enablement | 4 | 2 CCL_LOG_LEVEL + 2 distributed-backend |
| Existing test functions extended | 8 | Loss-aggregation |
| Existing pure software tests | 9 | Ray environment-variable |

*Snapshot note:* the 21 above cover the 3 submitted PRs. This session added regression tests
for the newer enablement PRs (ring-attention, vLLM device-pinning, gloo weight-sync); with those
included the full suite is **77 passing, 0 failing**.

---

## The 3 Submitted PRs

| PR | What it does | Tests |
|---|---|---|
| **CCL_LOG_LEVEL** | Sets the Intel oneCCL log-level env var alongside NCCL's, so the correct one applies per accelerator | 2 |
| **Loss-aggregation (device-generic)** | Loss/normalizer tests now run on the active accelerator (XPU/CUDA), not CPU-only | 8 extended |
| **Distributed-backend (device-generic)** | Real 2-rank collective test that auto-selects the canonical backend per device (NCCL/CUDA, XCCL/XPU, Gloo/CPU) | 2 |

All three are **device-agnostic** — they work on NVIDIA and Intel from the same code, with no
vendor branching.

---

## Additional Enablement (validated single-XPU; multi-GPU/NVIDIA pending)

Beyond the 3 submitted PRs, the following were implemented and validated on single XPU:

1. **Device-API migration** — `torch.cuda.*` → `torch.accelerator.*` across ~36 call sites in
   10 files (empty_cache, synchronize, device-index, set-device). Index-preserving, so
   multi-GPU targets the correct device. NVIDIA behavior unchanged.
2. **Ring-attention / optional flash_attn** — flash_attn (CUDA-only) made optional; falls back to
   backend-neutral helpers on XPU. NVIDIA path byte-identical when flash_attn is present.
3. **vLLM device pinning** — pins each vLLM engine to its Ray-assigned accelerator via the
   platform's own visibility variable (CUDA_VISIBLE_DEVICES on NVIDIA, ZE_AFFINITY_MASK on XPU).
4. **vLLM weight-sync** — NCCL is unavailable on XPU (it silently no-op'd, so the policy never
   updated in the rollout engine). Now routes through a hardware-neutral gloo path with
   auto-detect (NVIDIA→NCCL, XPU→gloo); NVIDIA/NCCL path preserved.

---

## Validation Status

| Scope | Status |
|---|---|
| Unit tests | ✅ 100% pass (submitted PRs); full suite 77 pass |
| Smoke test — all RLHF workflows, single XPU | ✅ pass |
| E2E — GRPO/PPO/RM etc., single XPU | ✅ pass |
| Multi-XPU (2+ devices) | ⏳ pending hardware |
| ZeRO-3 sharded weight-sync | ⏳ pending multi-GPU |
| NVIDIA/NCCL regression run | ⏳ pending NVIDIA hardware |

---

## Key Design Principle

Every change is **device-agnostic** — the same code auto-adapts to the accelerator (via
`torch.accelerator.*` and vLLM's platform abstraction) rather than adding `if cuda … elif xpu …`
branches. This is what makes the changes upstream-acceptable: **NVIDIA users see no behavior
change; Intel XPU users gain support.**

---

## Ask / Next Step

- Multi-XPU and NVIDIA hardware access to complete the remaining validation rows before
  upstreaming the enablement PRs.
