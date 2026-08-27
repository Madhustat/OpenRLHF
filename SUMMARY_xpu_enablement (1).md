# Enabling Intel XPU for OpenRLHF — Progress Update

## Quick recap

**At the last update:**

- **Unit tests: 100% pass on XPU** — all 12 XPU-runnable tests pass, with 0 failures and 0 blocked.
- **Three unit-test-backed PRs submitted upstream** for review.
- **OpenRLHF fully enabled on a single Intel XPU** for the validated RLHF workflows.

**Since the last update, we’ve progressed from a single-XPU prototype to full end-to-end validation on both single-XPU and multi-XPU configurations across all applicable training workflows.**

---

## Headline

- **Full RLHF loop runs end-to-end on Intel XPU** — validated on both a **single XPU** and
  **multiple XPUs**, spanning the complete Ray + vLLM + DeepSpeed stack.
- **Every training variant validated E2E** — GRPO, REINFORCE, RLOO, SFT, RM, DPO, and
  PPO-with-critic.
- **Weight synchronization proven both ways** — the shipping default (CPU-staged) runs every step
  across the full suite; a **direct XPU-to-XPU path** was additionally validated in a Docker
  container as the next-step optimization.
- **4 PRs** covering the core enablement — 2 submitted upstream, 2 in progress.
- **No NVIDIA regression by design** — one code path, no Intel-vs-NVIDIA branching; NVIDIA
  re-validated end-to-end.

---
# Enabling Intel XPU for OpenRLHF: Progress Update

## Quick Recap

**At the last update:**

- **XPU-runnable tests were passing at 100%:** All 12 tests passed, with no
  failures or blocked tests.
- **Three test-focused PRs had been submitted upstream** for review.
- **A single-XPU prototype was working** across the validated OpenRLHF
  workflows.

**Since the last update, we have progressed from a single-XPU prototype to
full end-to-end validation on both single-XPU and multi-XPU configurations
across all applicable training workflows.**

---

## Headline as of Today

- **The complete OpenRLHF pipeline runs end to end on Intel XPU:** The Ray,
  vLLM, and DeepSpeed stack has been validated on both single-XPU and
  multi-XPU configurations.

- **Seven training workflows have been validated:** GRPO, REINFORCE, RLOO,
  SFT, reward-model training, and DPO passed on single-XPU and multi-XPU
  configurations. PPO with critic was validated on multi-XPU.

- **Trainer-to-vLLM weight synchronization is working:** The CPU-staged Gloo
  path completed repeated synchronization across the validated workflows. A
  direct XPU-to-XPU path was also validated separately in a Docker container
  as a future performance optimization.

- **Five of seven planned PRs have been submitted upstream:** The remaining
  two PRs, covering vLLM device pinning and Gloo-based weight synchronization,
  are implemented and undergoing internal review.

- **NVIDIA regression validation is complete:** GRPO and PPO with critic were
  revalidated end to end on multi-GPU CUDA, with no regression observed in
  the tested workflows.

---

## The 4 PRs (the core enablement)

These are the pieces that were "work in progress" at the last update — now implemented, tested, and
moving upstream.

| PR | Delivers | Status |
|---|---|---|
| **Device-agnostic APIs** | Migrates the framework from CUDA-specific calls to PyTorch's device-agnostic accelerator APIs — the bulk of the enablement. | **Submitted** |
| **Optional flash-attn imports** | `flash_attn` is a CUDA-only package that OpenRLHF imported unconditionally, so the model stack wouldn't even import on XPU. Now falls back to equivalent PyTorch implementations when it's absent; the NVIDIA path is unchanged. | **Submitted** |
| **vLLM device pinning** | Pins each vLLM rollout worker to its assigned Intel device (XPU uses a different visibility variable than NVIDIA), so workers don't collide on device 0. | **In progress** |
| **vLLM weight sync (gloo)** | Routes vLLM weight synchronization through a CPU-staged gloo broadcast when NCCL/RCCL is unavailable (e.g. Intel XPU), so updated weights actually reach the rollout engine instead of being silently dropped. | **In progress** |

*Together with the three unit-test-backed PRs in the recap above, this brings the total to
**7 upstream PRs — 5 submitted, 2 in progress**.*

*Links for the submitted PRs and the roadmap issue:*

| Upstream | Title | Type | Status |
|---|---|---|:---:|
| [#1267](https://github.com/OpenRLHF/OpenRLHF/pull/1267) | test: propagate oneCCL log level to Ray workers | PR | Open |
| [#1268](https://github.com/OpenRLHF/OpenRLHF/pull/1268) | test: run loss aggregation checks on accelerators | PR | Open |
| [#1269](https://github.com/OpenRLHF/OpenRLHF/pull/1269) | test: add device-generic distributed backend smoke tests | PR | Open |
| [#1301](https://github.com/OpenRLHF/OpenRLHF/pull/1301) | use device-agnostic PyTorch accelerator APIs | PR | Open |
| [#1302](https://github.com/OpenRLHF/OpenRLHF/pull/1302) | make flash_attn utility imports optional | PR | Open |
| [#1303](https://github.com/OpenRLHF/OpenRLHF/issues/1303) | [Roadmap] OpenRLHF on Intel XPU | Issue | Open |

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
| PPO-with-critic | n/a | ✅ |

PPO-with-critic is multi-XPU only — actor, critic, and vLLM engine don't co-fit on a single device.

**NVIDIA control run:** GRPO and PPO-with-critic validated end-to-end on multi-GPU CUDA — no
regression from the device-agnostic changes.

---

## Weight synchronization — proven both ways

Keeping the policy weights and the vLLM rollout engine in sync every step is the crux of RLHF on
any accelerator. We validated two paths:

- **Default (shipping): CPU-staged gloo broadcast.** Portable, no vendor collective library
  required, ships with PyTorch. Runs correctly every step across the full E2E suite on 1 and 2 XPUs.
- **Optimization: direct XPU-to-XPU transfer.** Tested and working in a Docker container built on a
  newer PyTorch and vLLM XPU stack, running a clean multi-step GRPO job on two physical XPUs. This
  removes the CPU hop and is the natural next-step speedup once the supporting Intel wheels are
  generally available.

---

## Next steps

1. **Upstream the last 2 PRs** (device pinning + gloo weight sync), in dependency order behind the
   device-agnostic PR.
2. **End-to-end use cases** — identify suitable use cases and implement an E2E example for each
   training method.
