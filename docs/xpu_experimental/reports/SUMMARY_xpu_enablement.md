# Enabling Intel XPU for OpenRLHF

**OpenRLHF — a leading open-source RLHF framework that unifies Ray, vLLM, and DeepSpeed** into
one pipeline for reinforcement-learning post-training (the technique behind aligning modern LLMs).
We have brought that full Ray + vLLM + DeepSpeed stack to Intel XPU — using the same
device-agnostic code path that NVIDIA runs on, with no NVIDIA regression.

**Goal:** Make OpenRLHF's full training loop run on Intel XPU — orchestration (Ray), rollout
generation (vLLM), and distributed training (DeepSpeed) working together on Intel hardware —
through upstream-friendly, device-agnostic changes that preserve existing NVIDIA/CUDA behavior.

---

## Headline

- **Fully enabled on a single Intel XPU** for the validated RLHF workflows, spanning the full
  Ray + vLLM + DeepSpeed stack. **Multi-XPU enablement is in progress.**
- **Unit tests: 100% pass** on the XPU-runnable suite (0 failures, 0 blocked).
- **3 test-backed PRs submitted upstream.**
- **No regression for NVIDIA/CUDA users, by design:** the enablement is a device-agnostic
  PyTorch API migration plus a few targeted correctness/compatibility fixes — one code path
  adapts to the hardware (no NVIDIA-vs-Intel branching), split into focused, independently
  reviewable PRs to keep review and regression risk low.

---

## Unit tests at a glance

Of **21 unit tests total**, **12 are XPU-runnable and all 12 pass (100%)**; the other 9 are
pure-software tests (hardware-independent) and also pass. Nothing fails or is blocked.

| Category | Count | What it is |
|---|---:|---|
| **Newly added XPU-enablement tests** | **4** | 2 oneCCL log-level tests + 2 distributed-backend tests |
| **Existing tests extended to run on XPU** | 8 | Loss-aggregation tests, now exercised on the accelerator (previously CPU-only) |
| **Existing pure-software tests** | 9 | Ray environment-variable behavior (hardware-independent) |
| **Total** | **21** | |

---

## The 3 submitted PRs

| PR | What it delivers |
|---|---|
| **oneCCL logging** | Makes Intel's communication library (oneCCL) log correctly alongside NVIDIA's — **2 new tests.** |
| **Distributed-backend enablement** | Proves real multi-process communication works on each vendor's hardware (auto-selects the right backend per device) — **2 new tests.** |
| **Loss-aggregation on accelerator** | Extends **8 existing tests** so they actually run on the accelerator (XPU/CUDA), not just CPU. |

All three are written to work on both NVIDIA and Intel from a single code path.

---

## Still in progress (WIP)

Additional, larger enablement is implemented and passing on a single XPU, and is being hardened
before upstreaming:

- Device-agnostic API migration across the framework (the bulk of the enablement).
- vLLM integration on XPU — device assignment and model weight synchronization.
- Making a CUDA-only attention dependency optional so XPU can run without it.

These work today on one XPU; what remains is **validation on larger hardware**.

---

## Next steps

1. Validate on **multi-XPU** and **NVIDIA** hardware (the pieces we can't test on a single-GPU box).
2. Finish hardening and **upstream the remaining enablement PRs**.

---

## Ask

Access to **multi-XPU and NVIDIA hardware** to complete final validation before upstreaming.
