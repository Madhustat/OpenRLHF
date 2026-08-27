# Enabling Intel XPU for OpenRLHF — Progress Update

## Quick recap: where we were, where we are

OpenRLHF is a widely used open-source framework that combines Ray, vLLM and DeepSpeed into a single
pipeline for reinforcement-learning post-training of large language models. It is CUDA-first today.
Our work makes it run on Intel XPU through **one device-agnostic code path** — the same lines of
code NVIDIA runs — so there is no separate fork to maintain and no NVIDIA regression.

**At the last update**, we had unit-test coverage running on Intel XPU and submitted upstream, plus
a single-XPU prototype of the training loop.

**Now**, the full training loop runs end-to-end on Intel XPU on both a single device and multiple
devices, across all seven training variants. Weight synchronisation between training and rollout is
solved and validated. Direct XPU-to-XPU weight transfer has been tested and is working in a Docker
container. NVIDIA has been re-validated with no regression.

Seven upstream PRs are planned in total: **five are submitted and under review, and two are in
internal review before submission.**

---

## What we have completed

### The full RLHF training loop runs end-to-end on Intel XPU

Not unit tests — real Ray orchestration, real vLLM rollout generation, real DeepSpeed training,
with weights synchronised between them every step. Each variant was run for 10 steps.

| Training variant | Single XPU | Multi-XPU |
|---|:---:|:---:|
| GRPO | Pass | Pass |
| REINFORCE | Pass | Pass |
| RLOO | Pass | Pass |
| SFT | Pass | Pass |
| RM (reward model) | Pass | Pass |
| DPO | Pass | Pass |
| PPO with critic | Not applicable | Pass |

PPO with critic is multi-XPU only, because the actor, critic and vLLM engine do not fit together on
a single device.

**NVIDIA control run:** GRPO and PPO with critic were re-validated end-to-end on multi-GPU CUDA. No
regression from the device-agnostic changes.

### Weight synchronisation between training and rollout

Keeping the policy weights and the vLLM rollout engine in step on every training step is the core
problem of RLHF on any accelerator. Two paths were built and validated:

- **CPU-staged broadcast — the shipping default.** Portable, needs no vendor collective library,
  ships with PyTorch. This is what every end-to-end result above runs on.
- **Direct XPU-to-XPU transfer — tested and working in a Docker container.** Runs a clean
  multi-step GRPO job across two physical XPUs, removing the CPU hop. It runs inside the container
  image because no pip-installable pairing of the required PyTorch and vLLM versions has been
  released yet, so the container is how we ship it today.

### Five PRs submitted upstream and under review

| Upstream PR | What it delivers |
|---|---|
| [#1267](https://github.com/OpenRLHF/OpenRLHF/pull/1267) — Propagate oneCCL log level to Ray workers | Makes Intel collective-library logs visible inside Ray workers, the way NVIDIA's already are. Adds 2 tests. |
| [#1268](https://github.com/OpenRLHF/OpenRLHF/pull/1268) — Run loss-aggregation checks on accelerators | Extends 8 existing loss tests to run on the accelerator instead of CPU only. |
| [#1269](https://github.com/OpenRLHF/OpenRLHF/pull/1269) — Add device-generic distributed-backend smoke tests | Verifies real multi-process communication on whichever vendor backend is present. Adds 2 tests. |
| [#1301](https://github.com/OpenRLHF/OpenRLHF/pull/1301) — Use device-agnostic PyTorch accelerator APIs | Migrates the framework off CUDA-specific calls onto PyTorch's device-agnostic accelerator APIs. This is the bulk of the enablement. |
| [#1302](https://github.com/OpenRLHF/OpenRLHF/pull/1302) — Make flash_attn utility imports optional | `flash_attn` is CUDA-only and was imported unconditionally, so the model stack would not even load on XPU. Falls back to equivalent PyTorch implementations; the NVIDIA path is unchanged. |

Together these give **12 unit tests running and passing on Intel XPU** — 2 new in #1267, 8 existing
extended in #1268, and 2 new in #1269 — with no failures.

All seven PRs and the remaining work are tracked publicly under roadmap issue
[#1303](https://github.com/OpenRLHF/OpenRLHF/issues/1303).

---

## What we are working on now

Two PRs are in internal review before going upstream. Both are already implemented, and they are
what the end-to-end results above run on; what remains is the review and the submission.

| Change | What it delivers |
|---|---|
| vLLM device pinning | Pins each vLLM rollout worker to the Intel device Ray assigned it. XPU uses a different device-visibility variable than NVIDIA, so without this every worker collides on device 0. |
| vLLM weight synchronisation | Routes weight synchronisation through the CPU-staged broadcast when the NVIDIA collective library is unavailable, so updated weights actually reach the rollout engine on Intel hardware. |

---

## What is next

1. **Submit the two remaining PRs upstream**, in dependency order behind #1301.
2. **Identify suitable use cases and build an end-to-end example for each training method.**
3. **Move the direct XPU-to-XPU path onto a standard installation**, once compatible PyTorch and
   vLLM releases are available, so it no longer needs the container.
