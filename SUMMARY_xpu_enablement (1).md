# Enabling Intel XPU for OpenRLHF — Progress Update

## Quick recap

**At the last update:**

- **Unit tests: 100% pass on XPU** — all 12 XPU-runnable tests pass, with 0 failures and 0 blocked.
- **Three unit-test-backed PRs submitted upstream** for review.
- **A single-XPU prototype was working** across the validated OpenRLHF workflows.

**Since the last update, we've progressed from a single-XPU prototype to full end-to-end
validation on both single-XPU and multi-XPU configurations across all applicable training
workflows.**

---

## Headline

- **The complete OpenRLHF pipeline runs end to end on Intel XPU:** The Ray, vLLM, and DeepSpeed
  stack has been validated on both single-XPU and multi-XPU configurations.
- **Seven training workflows have been validated:** GRPO, REINFORCE, RLOO, SFT, reward-model
  training, and DPO passed on single-XPU and multi-XPU configurations. PPO with critic was
  validated on multi-XPU.
- **Trainer-to-vLLM weight synchronization is working:** The CPU-staged Gloo path completed
  repeated synchronization across the validated workflows. A direct XPU-to-XPU path was also
  validated separately in a Docker container as a future performance optimization.
- **Four PRs cover the core enablement:** Two have been submitted upstream, and the remaining two
  are implemented and in progress.
- **NVIDIA regression validation is complete:** GRPO and PPO with critic were revalidated end to
  end on multi-GPU CUDA, with no regression observed in the tested workflows.

---

## The Four Core Enablement PRs

These four contributions were work in progress at the last update. All four are now implemented
and tested. Two have been submitted upstream, and two are being prepared for submission.

| PR | Delivers | Status |
|---|---|:---:|
| **Device-agnostic APIs** | Migrates the framework from CUDA-specific calls to PyTorch's device-agnostic accelerator APIs — the bulk of the enablement. | **Submitted** |
| **Optional flash-attn imports** | `flash_attn` is a CUDA-only package that OpenRLHF imported unconditionally, so the model stack wouldn't even import on XPU. Now falls back to equivalent PyTorch implementations when it's absent; the NVIDIA path is unchanged. | **Submitted** |
| **vLLM device pinning** | Pins each vLLM rollout worker to the Intel XPU assigned by Ray. Intel XPU uses `ZE_AFFINITY_MASK` for device visibility, whereas NVIDIA uses `CUDA_VISIBLE_DEVICES`. Configuring the correct variable prevents multiple workers from selecting the same physical XPU.| **In progress** |
| **vLLM weight sync (gloo)** | Routes vLLM weight synchronization through a CPU-staged gloo broadcast when NCCL/RCCL is unavailable (e.g. Intel XPU), so updated weights actually reach the rollout engine instead of being silently dropped. | **In progress** |

Together with the three previously submitted test-focused PRs, the current contribution count is:

| Contribution | Count |
|---|:---:|
| Previously submitted test PRs | 3 |
| Submitted core-enablement PRs | 2 |
| Core PRs in progress | 2 |
| **Total planned PRs** | **7** |

*Links for the submitted PRs and the roadmap issue:*

| Upstream | Title | Type | Status |
|---|---|:---:|:---:|
| [#1267](https://github.com/OpenRLHF/OpenRLHF/pull/1267) | test: propagate oneCCL log level to Ray workers | PR | Open |
| [#1268](https://github.com/OpenRLHF/OpenRLHF/pull/1268) | test: run loss aggregation checks on accelerators | PR | Open |
| [#1269](https://github.com/OpenRLHF/OpenRLHF/pull/1269) | test: add device-generic distributed backend smoke tests | PR | Open |
| [#1301](https://github.com/OpenRLHF/OpenRLHF/pull/1301) | use device-agnostic PyTorch accelerator APIs | PR | Open |
| [#1302](https://github.com/OpenRLHF/OpenRLHF/pull/1302) | make flash_attn utility imports optional | PR | Open |
| [#1303](https://github.com/OpenRLHF/OpenRLHF/issues/1303) | [Roadmap] OpenRLHF on Intel XPU | Issue | Open |

The overall enablement effort and longer-term work are tracked through:
 [#1303](https://github.com/OpenRLHF/OpenRLHF/issues/1303)

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

## Weight synchronization: two validated paths

Keeping the policy weights and the vLLM rollout engine synchronized after each training update is
essential for RLHF. We validated two paths:

- **Current portable path: CPU-staged Gloo broadcast.** Gloo is provided by PyTorch and does not
  require a vendor-specific collective library. It completed weight synchronization throughout the
  validated single-XPU and multi-XPU workflows.
- **Performance optimization: direct XPU-to-XPU transfer.** This path was validated using a newer
  PyTorch and vLLM XPU stack in a Docker container, completing a multi-step GRPO run across two
  physical XPUs. It removes the CPU staging step and can be adopted when the required Intel
  software stack is available through the standard installation path.

---

## Next steps

1. **Upstream the last 2 PRs** (device pinning + gloo weight sync), in dependency order behind the
   device-agnostic PR.
2. **End-to-end use cases** — identify suitable use cases and implement an E2E example for each
   training method.
