# Enabling Intel XPU for OpenRLHF — Progress Update

**OpenRLHF — a leading open-source RLHF framework that unifies Ray, vLLM, and DeepSpeed** into
one pipeline for reinforcement-learning post-training (the technique behind aligning modern LLMs).
We have brought that full Ray + vLLM + DeepSpeed stack to Intel XPU — using the same
device-agnostic code path that NVIDIA runs on, with no NVIDIA regression.

Goal: Make OpenRLHF's full training loop run on Intel XPU — orchestration (Ray), rollout generation (vLLM), and distributed training (DeepSpeed) working together on Intel hardware — through upstream-friendly, device-agnostic changes that preserve existing NVIDIA/CUDA behavior.

**Since the last update:** we've moved from "3 test-backed PRs and a working single-XPU
prototype" to  **the full training loop validated end-to-end on Intel XPU — single *and* multi-XPU —
across every training variant**, plus a proven direct XPU-to-XPU weight-transfer path. The core
enablement is now landing upstream as **4 more PRs**.


## Headline or summary

- **Full RLHF loop runs end-to-end on Intel XPU** — validated on both a **single XPU** and
  **multiple XPUs**, spanning the complete Ray + vLLM + DeepSpeed stack.
- **Every training variant validated E2E with 10 steps** — GRPO, REINFORCE, RLOO, SFT, RM, DPO, and
  PPO-with-critic.
- **Weight synchronization proven both ways** — the shipping default (CPU-staged) runs every step
  across the full suite; a **direct XPU-to-XPU path** was additionally validated in an isolated
  container as the next-step optimization.
- **4 more PRs** covering the core enablement — 2 submitted upstream, 2 ready to submit.
- **No NVIDIA regression by design** — one code path, no Intel-vs-NVIDIA branching; NVIDIA
  re-validated end-to-end.

---

## The 4 new PRs (the core enablement)

These are the pieces that were "work in progress" last month — now implemented, tested, and
moving upstream.

| PR | Delivers | Status |
|---|---|---|
| **Device-agnostic APIs** | Migrates the framework from CUDA-specific calls to PyTorch's device-agnostic accelerator APIs — the bulk of the enablement. | **Submitted** |
| **Optional attention dep** | Makes a CUDA-only attention dependency optional, so the model stack imports and runs on XPU. | **Submitted** |
| **vLLM device pinning** | Pins each vLLM rollout worker to its assigned Intel device (XPU uses a different visibility variable than NVIDIA), so workers don't collide on device 0. | **Ready to submit** |
| **vLLM weight sync (gloo)** | Routes vLLM weight synchronization through a CPU-staged gloo broadcast when NCCL/RCCL is unavailable (e.g. Intel XPU), so updated weights actually reach the rollout engine instead of being silently dropped. | **Ready to submit** |

*Combined with the 3 PRs from last month (oneCCL logging, distributed-backend, loss-aggregation),
this brings the total to 7 upstream PRs.*

---

## End-to-end validation

The full training loop was exercised, not just unit-tested — real Ray orchestration, real vLLM
rollouts, real DeepSpeed training, weights synced every step.

| Training variant | Single XPU | Multi-XPU |
|---|:---:|:---:|
| GRPO | ✅ | ✅ |
| REINFORCE | ✅ | ✅ |
| RLOO | ✅ | ✅ |
| SFT | ✅ | ✅ |
| RM (reward model) | ✅ | ✅ |
| DPO | ✅ | ✅ |
| PPO-with-critic | — | ✅ |

**NVIDIA control run:** GRPO and PPO-with-critic validated end-to-end on multi-GPU CUDA — no
regression from the device-agnostic changes.

---

## Weight synchronization — proven both ways

Keeping the policy weights and the vLLM rollout engine in sync every step is the crux of RLHF on
any accelerator. We validated two paths:

- **Default (shipping): CPU-staged gloo broadcast.** Portable, no vendor collective library
  required, ships with PyTorch. Runs correctly every step across the full E2E suite on 1 and 2 XPUs.
- **Optimization: direct XPU-to-XPU transfer.** Validated in an isolated container (latest Intel
  runtime), running a clean multi-step GRPO job on two physical XPUs. This removes the CPU hop and
  is the natural next-step speedup once the supporting Intel wheels are generally available.

---

## Where we stand at a glance

```
Last month:   3 test PRs submitted  |  single-XPU prototype working
This month:   full E2E on 1 + N XPUs, all variants  |  4 core PRs (2 up, 2 ready)
              direct XPU-to-XPU weight sync proven  |  NVIDIA re-validated, no regression
```

---

## Next steps

1. **Upstream the last 2 PRs** (device pinning + gloo weight sync), in dependency order behind the
   device-agnostic PR.
2. **Broaden hardware coverage** — larger multi-XPU and multi-node scale.
3. **Land the direct XPU-to-XPU path** as a follow-up optimization once native Intel wheels ship.
