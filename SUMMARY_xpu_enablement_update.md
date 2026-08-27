# Enabling Intel XPU for OpenRLHF — Progress Update

## Quick recap

- **Unit tests: 100% pass on XPU** — all 12 XPU-runnable tests pass, with 0 failures and 0
  blocked (2 new in #1267, 8 existing extended in #1268, 2 new in #1269).
- **5 of the planned 7 PRs are submitted upstream**, all open for review, plus a roadmap issue
  tracking the overall effort.

| Upstream | Title | Type | Status |
|---|---|---|:---:|
| [#1267](https://github.com/OpenRLHF/OpenRLHF/pull/1267) | test: propagate oneCCL log level to Ray workers | PR | Open |
| [#1268](https://github.com/OpenRLHF/OpenRLHF/pull/1268) | test: run loss aggregation checks on accelerators | PR | Open |
| [#1269](https://github.com/OpenRLHF/OpenRLHF/pull/1269) | test: add device-generic distributed backend smoke tests | PR | Open |
| [#1301](https://github.com/OpenRLHF/OpenRLHF/pull/1301) | use device-agnostic PyTorch accelerator APIs | PR | Open |
| [#1302](https://github.com/OpenRLHF/OpenRLHF/pull/1302) | make flash_attn utility imports optional | PR | Open |
| [#1303](https://github.com/OpenRLHF/OpenRLHF/issues/1303) | [Roadmap] OpenRLHF on Intel XPU | Issue | Open |

**Since the last update:** we've moved from a working single-XPU prototype to **the full training
loop validated end-to-end on Intel XPU — single *and* multi-XPU — across every training variant**,
plus a proven direct XPU-to-XPU weight-transfer path. The core enablement is now landing upstream as 4 PRs — 2 submitted and under review, 2 still to be submitted.


## Headline

- **Full RLHF loop runs end-to-end on Intel XPU** — validated on both a **single XPU** and
  **multiple XPUs**, spanning the complete Ray + vLLM + DeepSpeed stack.
- **Every training variant validated E2E with 10 steps** — GRPO, REINFORCE, RLOO, SFT, RM, DPO, and
  PPO-with-critic.
- **Weight synchronization proven both ways** — the shipping default (CPU-staged) runs every step
  across the full suite; a **direct XPU-to-XPU path** was additionally validated in an isolated
  container as the next-step optimization.
- **4 more PRs** covering the core enablement — 2 submitted upstream, 2 are in progress.
- **No NVIDIA regression by design** — one code path, no Intel-vs-NVIDIA branching; NVIDIA
  re-validated end-to-end.

---

## The 4 new PRs (the core enablement)

These are the pieces that were "work in progress" last month — now implemented, tested, and
moving upstream.

| PR | Delivers | Status |
|---|---|---|
| **Device-agnostic APIs** | Migrates the framework from CUDA-specific calls to PyTorch's device-agnostic accelerator APIs — the bulk of the enablement. | **Submitted** |
| **Optional flash-attn imports** | `flash_attn` is a CUDA-only package that OpenRLHF imported unconditionally, so the model stack wouldn't even import on XPU. Now falls back to equivalent PyTorch implementations when it's absent; the NVIDIA path is unchanged. | **Submitted** |
| **vLLM device pinning** | Pins each vLLM rollout worker to its assigned Intel device (XPU uses a different visibility variable than NVIDIA), so workers don't collide on device 0. | **Ready to submit** |
| **vLLM weight sync (gloo)** | Routes vLLM weight synchronization through a CPU-staged gloo broadcast when NCCL/RCCL is unavailable (e.g. Intel XPU), so updated weights actually reach the rollout engine instead of being silently dropped. | **In Progress** |

*Combined with the 3 PRs from last month, this brings the total to 7 upstream PRs.*

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

PPO-with-critic is multi-XPU only — actor, critic, and vLLM engine don't co-fit on a single device.

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
2.  **End-to-end use cases** — identify suitable use cases and implement an E2E example for each
   training method.
3. **Land the direct XPU-to-XPU path** as a follow-up optimization once native Intel wheels ship.
