# This is a WORKING branch, not a PR branch

`openrlfh_fresh_multixpu` is the multi-XPU **working repo**: the exact tree that produced the
recorded results in `tests/multixpu/recorded_results/`. It is deliberately NOT PR-ready.

Base: upstream `dc2a7ad3` (0 commits behind at the time of commit).

## Four local-only changes are INCLUDED on purpose

They are kept so this branch reproduces its own results. **Strip all four before opening any
pull request.**

| # | What | Where | Why it is here | Why it must not be PR'd |
|---|---|---|---|---|
| 1 | `OPENRLHF_WEIGHT_PROBE` instrumentation | `trainer/ray/ppo_actor.py`, `trainer/ray/vllm_worker_wrap.py`, `cli/train_ppo_ray.py` | Logs parameter checksums on both sides of the broadcast, so a run can be *proven* to have transferred weights rather than merely not crashed. Weight sync on a non-CUDA accelerator can fail **silently**: `PyNcclCommunicator` sets `disabled=True` when NCCL is unavailable and every `broadcast()` then returns without moving anything. That actually happened on this project. | Diagnostic scaffolding, not a feature. Gated on `OPENRLHF_WEIGHT_PROBE=1`, default `0`, so production never executes it. |
| 2 | `eos_indices.to(values.device)` | `models/model.py` | Reward-model forward gathers across two devices when `device_map="auto"` splits the model. | Already proposed upstream: issue **#938**, fix posted in a comment by the reporter. Not ours to re-propose. |
| 3 | `_init_weights` restoring `mean`/`std` | `models/model.py` | `mean`/`std` are `persistent=False`, so `device_map` loading leaves them uninitialised (`std=0.0` -> `reward=(r-0)/0=inf`). | Already proposed upstream: open PR **#1343**. Not ours to re-propose. |
| 4 | `self.mean.to(reward.device)` / `values.device` | `models/model.py` | #938's one-liner only moves the crash one line down: `reward` sits on the score head's device while `mean`/`std` stay on the primary device, because accelerate never dispatches them (it warns `device_map keys do not match any submodules: ['mean','std']`). | This one is genuinely new and unreported. The right venue is a comment on #938 or #1343, not a patch bundled into an XPU-enablement PR. |

Without #2-#4, `up_serve_rm` cannot run its full 2-device configuration: it must be pinned to a
single device with `--reward.normalize_enable` dropped. That is why they are committed here —
removing them would make this branch stop reproducing its own recorded result.

## Two changes that are LOAD-BEARING — do NOT strip these

| What | Where | Without it |
|---|---|---|
| `overlap_comm=False` for ZeRO-3 | `utils/deepspeed/deepspeed_utils.py` | ZeRO-3 SIGSEGVs in oneCCL's progress thread at **any** rank count. The 80-cell matrix went 0/18 -> 18/18 at ZeRO-3 because of this line. |
| FusedAdam-capability fallback | `utils/deepspeed/deepspeed_utils.py` | `--ds.enable_sleep` on ZeRO-3 crashes on DeepSpeed's FusedAdam-only assert wherever FusedAdam cannot JIT-build (no `icpx` here). |

## Reverting the four, when the time comes

```bash
# 2, 3 and 4 are the ONLY local changes in model.py, so one command reverts all three:
git checkout -- openrlhf/models/model.py

# 1: remove the OPENRLHF_WEIGHT_PROBE blocks from the three files above, and delete the
#    3 probe tests from tests/test_ppo_zero3_fixes_suite.py (suite goes 9 -> 6 tests).
```

A `pre-commit` / `pre-push` guard in `.git/hooks/` detects all four and refuses to commit or push
with a loud banner. It was overridden once, deliberately, for this working branch:
`OPENRLHF_ALLOW_LOCAL_PATCHES=1`. Do not override it for a PR branch.

## Where the tests are

Everything under `tests/`:

```
tests/HOW_TO_RUN_multixpu.md          how to run all three suites + environment gotchas
tests/MULTIXPU_TEST_PLAN.md           all 94 rows: intent, GPU-0/GPU-1, sleep, colocation, result
tests/MULTIXPU_ALGO_COVERAGE.md       53 algorithms/methods, proven or not
tests/MULTIXPU_CASE_TABLE.md          what each extended case actually checks
tests/test_e2e_suite_multigpu_extended.sh    57 cases (per-feature-flag coverage)
tests/test_e2e_suite_multigpu.sh             28 cases (base suite)
tests/run_multigpu_extended_with_rca.py      suite + automatic root-cause + retry
tests/multixpu/run_upstream_bench.sh         14 cases (upstream examples/scripts)
tests/multixpu/run_gloo_matrix.py            80 cases (topology x sleep x ZeRO stage)
tests/multixpu/helpers/                      7 operational scripts
tests/multixpu/recorded_results/             the outcomes, so numbers are checkable
```

Result at time of commit: **87 PASS, 3 FAIL, 2 SKIP, 1 N/A, 1 parked** of 94 tracked behaviours.
