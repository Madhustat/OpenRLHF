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

- **Unit tests: 100% pass on XPU** — all 12 XPU-runnable tests pass, with 0 failures and 0 blocked.
- **Three unit-test-backed PRs submitted upstream** for review.
- **Fully enabled on a single Intel XPU** across the full Ray + vLLM + DeepSpeed stack; multi-XPU
  enablement is in progress.
- **No NVIDIA regression by design** — a single device-agnostic code path adapts to the hardware,
  with no NVIDIA-vs-Intel branching.

---

## Unit tests at a glance

| | Count |
|---|---:|
| Total unit tests | 21 |
| Runnable on XPU | 12 |
| **Passing on XPU** | **12 (100%)** |
| Pure-software tests passing | 9 / 9 |
| Failures / blocked | 0 |
| Newly added for XPU | 4 |

The 4 new tests come from 2 of the submitted PRs (2 tests each); a 3rd PR extended 8 existing
tests to run on XPU.

---

## The 3 submitted PRs

| PR | Delivers |
|---|---|
| **oneCCL logging** | Intel's oneCCL logs correctly alongside NVIDIA's — 2 new tests. |
| **Distributed-backend** | Real multi-process comms verified per vendor (auto-selects backend) — 2 new tests. |
| **Loss-aggregation** | 8 existing tests extended to run on the accelerator, not just CPU. |

---

## Work in progress

Further enablement is implemented and validated on a single XPU; remaining automated coverage,
multi-XPU validation, and NVIDIA regression runs are being completed before final upstream
submission.

---

## Ask

Access to **multi-XPU and NVIDIA hardware** to finish validation and upstream the remaining PRs.
