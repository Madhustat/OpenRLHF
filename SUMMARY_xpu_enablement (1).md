# Enabling Intel XPU for OpenRLHF — Progress Update

## Quick recap

**At the last update:**

- **Unit tests: 100% pass on XPU** — all 12 XPU-runnable tests pass, with 0 failures and 0 blocked.
- **Three unit-test-backed PRs submitted upstream** for review.
- **A single-XPU prototype was working** across the validated OpenRLHF workflows.

**Since the last update, we've progressed from a single-XPU prototype to full end-to-end
validation on both single-XPU and multi-XPU configurations across applicable training
workflows.**

---

## Headline

- **The complete OpenRLHF pipeline runs end to end on Intel XPU:** The Ray, vLLM, and DeepSpeed
  stack has been validated on both single-XPU and multi-XPU configurations.
- **Six training workflows have been validated:** GRPO, REINFORCE, RLOO, SFT, reward-model
  training, and DPO passed on single-XPU and multi-XPU configurations. 
- **Trainer-to-vLLM weight synchronization is working:** The CPU-staged Gloo path completed
  repeated synchronization across the validated workflows.
- **Four PRs cover the core enablement:** Two have been submitted upstream, and the remaining two
  are implemented and submission is in progress'.
- **NVIDIA regression validation is complete:** RL workflows were revalidated end to
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
| **vLLM weight sync (Gloo)** | Routes vLLM weight synchronization through a CPU-staged Gloo broadcast when NCCL/RCCL is unavailable (e.g. Intel XPU), so updated weights actually reach the rollout engine instead of being silently dropped. | **In progress** |

Together with the three previously submitted test-focused PRs, the current PR count is: 7

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

| Training workflow | Single XPU | Multi-XPU |
|---|:---:|:---:|
| GRPO | ✅ | ✅ |
| REINFORCE | ✅ | ✅ |
| RLOO | ✅ | ✅ |
| SFT | ✅ | ✅ |
| RM (reward model) | ✅ | ✅ |
| DPO | ✅ | ✅ |

**NVIDIA control run:**  RL workflow validated end-to-end on multi-GPU CUDA — no
regression from the device-agnostic changes.

---

## Weight synchronization:

After each training update, the latest policy weights must be transferred to
the vLLM rollout engine so that subsequent rollouts use the updated model.

We enabled a portable CPU-staged Gloo broadcast path that works with the
currently supported OpenRLHF software stack. Gloo is provided by PyTorch and
does not require a vendor-specific collective library.

The Gloo path completed repeated weight synchronization throughout the
validated single-XPU and multi-XPU workflows.

---

## Next steps

1. **Upstream the last 2 PRs** (device pinning + Gloo weight sync), in dependency order behind the
   device-agnostic PR.
2. **Explore direct XPU-to-XPU weight synchronization using the latest compatible software stack**
3. **End-to-end use cases** — identify suitable use cases and implement an E2E example for each
   training method.
