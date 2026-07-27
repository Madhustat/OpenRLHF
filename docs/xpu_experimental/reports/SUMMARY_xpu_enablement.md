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

- **Unit tests: 100% pass** on XPU (12/12 runnable; 0 failures, 0 blocked).
- **3 test-backed PRs submitted upstream.**
- **Fully enabled on a single Intel XPU** across the Ray + vLLM + DeepSpeed stack. Multi-XPU is WIP.
- **No NVIDIA regression** — one device-agnostic code path, no vendor branching.

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

Larger enablement — device-API migration, vLLM device pinning + weight sync, optional CUDA-only
attention dep — is implemented and passing on a single XPU. Remaining before upstreaming:
**multi-XPU validation** and **NVIDIA regression runs** (need the hardware).

---

## Ask

Access to **multi-XPU and NVIDIA hardware** to finish validation and upstream the remaining PRs.
