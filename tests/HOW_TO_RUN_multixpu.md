# OpenRLHF MULTI-XPU test suites — 2x Intel Arc Pro B70

Validation of OpenRLHF across **two** Intel XPUs. The single-XPU baseline is separate:
branch `experimental-e2e-baseline-1xpu` of Madhustat/OpenRLHF (50 cases).

Everything needed to re-run the validation of OpenRLHF on two Intel Arc Pro B70 XPUs,
plus the recorded results. Three independent suites, each answering a different question.

| Suite | Cases | Question it answers | Recorded result |
|---|---|---|---|
| `01-upstream-scripts-bench` | 14 | Does every script in upstream `examples/scripts/` run here? | **14 / 14 PASS** |
| `02-gloo-matrix-80` | 80 | Does PPO survive every placement x sleep x ZeRO-stage combination? | **63 / 63 passable cells PASS** |
| `03-multigpu-extended` | 57 + 28 | Is every individual feature flag exercised? | **87 PASS, 3 FAIL, 2 SKIP, 1 N/A, 1 parked** |

Combined: **87 of 94 tracked behaviours pass.** See `docs/TWO_GPU_TEST_PLAN.md` for the full
per-case table with GPU-0/GPU-1 placement, sleep and colocation columns.

---

## Case inventory — verified present, not claimed

Counted directly from the shipped scripts:

| Suite | Script | Cases |
|---|---|---|
| 01 | `run_upstream_bench.sh` | 12 helper-invoked + 2 custom = **14** |
| 02 | `run_gloo_matrix.py` | 24 scenarios (4 invalid) x 4 ZeRO stages = **80 executable** |
| 03 | `test_e2e_suite_multigpu_extended.sh` | 51 flag-driven + 2 dependency-gated + 4 hand-rolled = **57** |
| 03 | `test_e2e_suite_multigpu.sh` (base) | 19 PPO + 8 supervised + 1 VLM = **28** |

Every case ID referenced in `docs/TWO_GPU_TEST_PLAN.md` was cross-checked against these four
scripts: **0 missing**. The 8 cases that were "yet to test" in earlier drafts are real cases in
the base suite (`rloo_nocolo`, `dr_grpo_nocolo`, `grpo_ema`, `grpo_overlong_penalty`,
`grpo_reward_norm`, `ppo_ref_model_nocolo`, `dpo_ipo`, `dpo_cdpo`) and all now PASS.

## Prerequisites

These suites do NOT create the environment; they assume one already works.

```
Hardware   2x Intel Arc Pro B70 (32.6 GiB each as reported by xpu-smi)
Python     3.13
torch      2.13.0+xpu
vLLM       0.27.2.dev0+xpu   (source build; 0.29 is NOT validated here)
DeepSpeed  0.19.1
Ray        2.55.0
transformers 5.16.0
venv       /home/sdp/venvs/openrlhf-xccl-auto-detect-213
repo       /home/sdp/madhu/OpenRLHF-fresh   (latest upstream + XPU changes)
```

Two environment facts that will cost you hours if missed:

1. **`LD_LIBRARY_PATH` must put the venv lib first.** The base miniforge `lib` ships an older
   `libur_loader`, and torch 2.13's `libsycl.so.9` then fails with
   `undefined symbol: urDeviceWaitExp`. Every launcher here already sets it.
2. **`OPENRLHF_DS_TORCH_ADAM=1` is required** on a box without `icpx`: DeepSpeed cannot JIT-build
   FusedAdam, so it must fall back to `torch.optim.AdamW`.

A third, measured on this hardware:

3. **`CCL_*=direct` must stay ON for any 2-device run.** With oneCCL's default topology-aware
   algorithms, every `actor_world_size=2` case dies with
   `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`. A/B on 2026-09-19:
   X12-Z2 FAIL 0/5 and X12-Z3 FAIL 0/5 with defaults, PASS 5/5 for both with `direct`. It is the
   slower host-staged path, but the fast path does not work here — it is working-vs-crashing,
   not fast-vs-slow. `run_gloo_matrix.py` sets it; `GLOO_MATRIX_CCL_DEFAULT=1` turns it off if
   you want to re-measure on newer software.

---

## 1. Upstream scripts bench — 14 cases, ~50 min

Adapts every runnable script in upstream `examples/scripts/` to 2 XPUs and a small model,
while keeping each script's distinctive algorithm configuration verbatim.

```bash
cd 01-upstream-scripts-bench
REPO=/home/sdp/madhu/OpenRLHF-fresh ./run_upstream_bench.sh            # all 14
REPO=/home/sdp/madhu/OpenRLHF-fresh ./run_upstream_bench.sh up_ppo_gae # one case
```

PASS requires more than exit 0: RL cases need at least one `Global step` AND a finite
`policy_loss`; PPO+critic additionally a finite `critic_loss`; supervised cases a finite loss.
Any NaN/Inf fails the case. Results land in `results_<timestamp>/summary.txt`.

Covers: PPO/GAE, DAPO, ProRL-v2, REINFORCE++, FlashREINFORCE, agent-async, VLM, SFT, SFT+LoRA,
RM, DPO, remote reward function, reward-model HTTP server, ZeRO->universal checkpoint conversion.

## 2. Gloo matrix — 80 PPO cases, ~5 h

The memory/placement sweep: **20 valid topologies x 4 sleep modes x 4 ZeRO stages**.
This is the suite that proves the ZeRO-3 `overlap_comm` fix.

```bash
cd 02-gloo-matrix-80
export GLOO_MATRIX_REPO=/home/sdp/madhu/OpenRLHF-fresh
python run_gloo_matrix.py --list                      # print the matrix, run nothing
python run_gloo_matrix.py                             # all 80
python run_gloo_matrix.py --only X12 --stages 3        # one cell
python run_gloo_matrix.py --resume run_<id>            # continue an interrupted sweep
python run_full_matrix_with_retry.py                   # sweep + auto root-cause + retry failures
python matrix_table.py results/run_<id>                # render the stage-column table
```

Result on the current tree: **ZeRO-1, ZeRO-2 and ZeRO-3 all 18/18**. ZeRO-3 was **0/18** before
the `overlap_comm=False` fix, so this suite is that fix's evidence.

Expected non-passes, by design, not defects:
- `X5-X8` (16 cells) — never run: critic world size cannot exceed actor world size on 2 devices
- `X1,X2` x 4 stages (8 cells) — blocked: upstream makes async and vLLM sleep mutually exclusive
- 9 stage-0 cells with DeepSpeed sleep ON — architecturally unsupported: stage 0 builds
  `FP16_UnfusedOptimizer`, which has no `offload_states()` at all. 9/9 correlation: every
  stage-0 row with DS sleep On fails, every DS-sleep-Off row passes.
- `X12-Z0` needs `gpu_memory_utilization >= 0.40`; a per-case override is in the script with the
  measured ladder (-5.40 / -2.98 / +0.05 / +3.08 GiB of KV cache at 0.22 / 0.30 / 0.40 / 0.50).

## 3. Multi-GPU extended suite — 57 cases, ~4.5 h

Per-feature-flag coverage. Complements the 28-case base suite shipped in the same folder.

```bash
cd 03-multigpu-extended
export PYTHON=/home/sdp/venvs/openrlhf-xccl-auto-detect-213/bin/python
export RAY=/home/sdp/venvs/openrlhf-xccl-auto-detect-213/bin/ray
export LD_LIBRARY_PATH=/home/sdp/venvs/openrlhf-xccl-auto-detect-213/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/home/sdp/madhu/OpenRLHF-fresh
export OPENRLHF_DS_TORCH_ADAM=1

python prepare_e2e_data.py                             # generate GSM8K prompts + SFT parquet ONCE
bash test_e2e_suite_multigpu.sh                        # base 28 cases
bash test_e2e_suite_multigpu_extended.sh               # extended 57 cases
bash test_e2e_suite_multigpu_extended.sh mg_gspo       # one case
python run_multigpu_extended_with_rca.py               # suite + auto root-cause + retry
```

Two things this suite gets right that cost real time to learn:

- **Per-case timeout.** Every training invocation is wrapped in `timeout` (`T_RL=600s` against a
  measured ~270 s healthy case). Without it, one hang blocked an entire run for 2 h 46 min after
  only 2 of 55 cases. `rc=124` is recorded as TIMEOUT, distinct from a crash.
- **GPU clearing before AND after every case**, waiting until both devices actually drop below
  600 MiB rather than assuming they drained. A straggler holding memory makes the *next* case fail
  with a spurious OOM, which reads as a real result.

### Placement patterns used (important when reading results)

| | GPU 0 | GPU 1 | Sleep | Colocation |
|---|---|---|---|---|
| P1 separated RL | Actor (DeepSpeed) | vLLM engine | Off / Off | none |
| P2 colocated RL | Actor + ref/critic/reward + vLLM | shared | On / On | `colocate_all` |
| P3 sharded actor | Actor rank 0 + engine 0 | Actor rank 1 + engine 1 | On / On | `colocate_all` |
| P4 supervised | Trainer, single process | **IDLE** | n/a | n/a |

**P4 is stated explicitly on purpose.** The supervised trainers run as ONE process
(`python -m openrlhf.cli.<trainer>`, no deepspeed/torchrun launcher, zero rank markers in the
logs => `world_size=1`). They exercise compute, loss and data paths on one device; they do NOT
test distribution or weight transfer, and ZeRO-3 in a P4 row proves the code path runs, not that
it shards. Anyone auditing these results should see that without having to ask.

RL rows do verify the transfer: measured ~290 parameters broadcast actor->engine per step over
gloo with CPU staging (1450 `update weight:` lines for 5 steps), sync process group
`world_size=2`. PASS needs at least one `Global step`, so a silent no-op sync cannot pass.

---

## shared/ — operational helpers

| Script | Use |
|---|---|
| `cleanup_stuck.sh` | Kill a stuck run and drain both XPUs |
| `check_suite.sh` | Progress of the extended suite + stall warning if no I/O for >15 min |
| `run_one_case.sh <case>` | Run a single extended-suite case |
| `validate_fixes.sh` | Re-run only the cases whose config changed |
| `run_gaps.sh` | Run only the "yet to test" cases (12 of them, ~55 min) |
| `launch_extended_suite.sh` | Launch the extended suite detached |
| `inventory.sh` | Branch/HEAD/dirty state of every OpenRLHF tree on the box |

**Why these live in files rather than being typed inline:** the suites call
`pkill -f` on patterns like `openrlhf.cli`, `ray::` and `VLLM::EngineCore` between cases. If the
invoking shell's own command line contains one of those strings, it gets killed mid-run. A
script's command line is just its path, so it is immune. This cost several aborted runs before
it was understood.

## docs/

| File | Contents |
|---|---|
| `TWO_GPU_TEST_PLAN.md` | All 94 rows: intent, GPU-0/GPU-1, sleep, colocation, result, failure reason |
| `ALGO_COVERAGE.md` | 53 algorithms/methods, each marked PROVEN or not, with the case that proves it |
| `CASE_TABLE_DETAILED.md` | The 56 extended cases with what each one actually checks |
| `MASTER_COVERAGE.py/.md` | Generator + output: 133 testable behaviours resolved against recorded results |

`MASTER_COVERAGE.py` is re-runnable: point it at fresh result directories and it recomputes
coverage from real outcomes rather than from assertions.

---

## Known failures, with causes

| Case | Cause | Whose |
|---|---|---|
| `mg_ppo_4bit` | bitsandbytes packs 4-bit weights as `uint8`; `vllm_worker_wrap.py:41` asserts every param matches the model dtype (`bf16`). The assert fires inside a Ray remote on ONE side of the gloo collective, so the engine never joins and the actor blocks forever — a clean error becomes an unbounded hang. Only 2 of ~290 params synced. `mg_sft_4bit` PASSES, so quantisation itself is fine; there is simply no dequantise-before-broadcast step. | OpenRLHF, not XPU-specific |
| `mg_sync_with_ray` | Separated placement -> `the new group's world size should be less or equal to the world size set by init_process_group`. With `colocate_all` -> `The collective APIs shall be only used inside a Ray actor or task`. Ray's collective group must be created inside a Ray actor and is not on this path. | Upstream, looks unmaintained |
| `mg_determinism` | Two RL runs with a fixed seed and `--train.full_determinism_enable` diverge at step 1. **Localised**: `mg_determinism_sft` is bit-identical across two runs, so the XPU compute kernels and DeepSpeed ARE deterministic — the non-determinism is in the vLLM rollout/sampling path. | vLLM rollout path |

Skipped for absent optional packages (not defects): `mg_liger` (`liger_kernel`),
`mg_ring_attn2` (`ring_flash_attn`).

Parked: `mg_moe_grpo` (MoE reinforcement learning). GraniteMoE fails at the initial weight sync
with `KeyError: layers.0.block_sparse_moe.router.weight` because transformers' runtime parameter
names diverge from the checkpoint names vLLM's loader expects; a tiny Qwen2Moe clears the names
then hits a shape mismatch that cannot be separated from a toy-checkpoint artefact. The smallest
real Qwen2Moe is 14.3B (~29 GB bf16), which does not fit on 2x16 GB alongside vLLM and the
optimizer, so this stays UNPROVEN rather than broken.

## Two upstream bugs found by these suites

1. **transformers GraniteMoE ignores `output_router_logits`.** It returns `router_logits=None`
   and `aux_loss` as a plain `int 0`, so transformers' own code dies at
   `modeling_granitemoe.py:644` on `aux_loss.to(loss.device)`. Verified that Qwen2Moe and Mixtral
   both return a proper Tensor. OpenRLHF's `sft_trainer.py:202` `.item()` crash is the same defect
   seen downstream (its guard tests `aux_loss_coef > 1e-8`, not the value's type;
   `rm_trainer.py:188` repeats the pattern).
2. **`serve_rm` is broken on any multi-GPU box.** `device_map="auto"` is accelerate's *balanced*
   strategy: it splits the model across every visible GPU even when it fits on one, landing the
   `score` head on device 1 while `model.device` stays device 0. `model.py:235` then gathers
   across two devices. Reported upstream as issue **#938** (32B RM, `cuda:7` vs `cuda:0`), closed
   2025-04-07 as "completed" with the fix only in a comment — never merged. Not XPU-specific.
   A second, separate defect: `mean`/`std` are registered `persistent=False`, so `device_map`
   loading leaves them uninitialised (`std=0.0` -> `reward=(r-0)/0=inf`); open upstream PR **#1343**.
