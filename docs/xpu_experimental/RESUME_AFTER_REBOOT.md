# RESUME POINT — Read this first after any machine restart

**Last updated:** 2026-07-09, ~14:50 (before a planned reboot to clear a stuck GPU process)
**If your IP address changed:** none of the file paths below depend on IP — everything is
local paths under `/home/dut7054/madhu/`. The only place an IP is used is inside Ray startup
commands (`ray start --head ...` auto-detects the current machine's IP; no manual IP config
needed) and it's mentioned once below purely as a historical log detail, not something to
reuse.

**Tell a fresh Claude Code session to read this file plus [[project_openrlhf_xpu_phase1]] in
memory (if memory is available) to get full context. If memory is NOT available (e.g. new
machine/environment), this file plus the other reports in
`~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/` are self-contained.**

---

## 0. IMMEDIATE ACTION NEEDED (why you're rebooting)

A process got stuck holding the Intel GPU driver in an unkillable state (`kill -9` had no
effect for 18+ minutes, `xpu-smi` itself hung trying to query the GPU). See section 6 for the
full incident. **After reboot, verify the GPU is usable again before doing anything else:**
```bash
xpu-smi discovery -d 0
```
This should return promptly. If it hangs again, something deeper is wrong (see section 6's
"if this recurs" note) — don't just retry the same training command.

**Also: check for leftover Ray/vLLM processes and clean up before starting anything new:**
```bash
ps aux | grep -iE "ray::|EngineCore|PolicyModelActor|ReferenceModelActor|RolloutRayActor"
rm -rf /tmp/ray
```

---

## 1. What this whole effort is

**Goal:** validate that OpenRLHF (an open-source RLHF training framework — Ray for distributed
orchestration, vLLM for fast rollout generation, DeepSpeed for training, Transformers for model
loading — "the four pillars") works correctly on Intel XPU (Arc Pro B70 GPU), not just NVIDIA.
Fix real bugs found along the way. Then prove it with an actual GRPO training run on a real
benchmark (GSM8K, a math word-problem dataset) showing a genuine accuracy lift, comparable in
spirit to what you'd see on NVIDIA hardware.

**Three working directories exist** (see `project_openrlhf_xpu_phase1.md` in memory for full
detail if available):
- `~/madhu/work1_with customfunction/` — first attempt, uses a custom vendored file instead of
  transformers' built-in flash-attention replacement. Reference only, not active.
- `~/madhu/work2_with_transformers_native/` — **the active environment, everything below refers
  to this one.** venv at `OpenRLHF/venv-tf`.
- `~/madhu/work3_verl_style/` — a one-time comparison build (tested a colleague's Docker-based
  approach). Frozen, not actively maintained, no bearing on what's below.

---

## 2. Everything fixed in OpenRLHF's code so far (all in work2, all UNCOMMITTED — see section 7)

17 files modified from upstream OpenRLHF, all still applied on disk as of this writing
(confirmed via `git diff --stat` in `~/madhu/work2_with_transformers_native/OpenRLHF/`):

**Pattern A — device portability (`torch.cuda.*` → `torch.accelerator.*`)**, across
`ring_attn_utils.py`, `ppo_trainer.py`, `ppo_trainer_async.py`, `replay_buffer.py`,
`samples_generator.py`, `trainer/ray/__init__.py`, `launcher.py`, `ppo_actor.py`,
`ppo_critic.py`, `utils.py`, `vllm_worker_wrap.py`, `deepspeed.py`, `deepspeed_utils.py`,
`distributed_util.py`, `loss_utils.py`. This is PyTorch's own official cross-vendor API — works
identically on NVIDIA, no behavior change there.

**Pattern B — genuine bugs, not XPU-specific** (would affect any platform):
- `train_ppo_ray.py`: `max_len` `UnboundLocalError` when `--vllm.num_engines 0`.
- `train_rm.py`: eval-set fallback rounds to 0 rows for small datasets, crashes with a
  misleading error.
- `ppo_actor.py`: a guard bug that ran `import vllm` even when vLLM was disabled.

**Pattern C — replaced `flash_attn` dependency** (`ring_attn_utils.py`) with
transformers-native equivalents (`transformers.modeling_flash_attention_utils`), since
`flash_attn` doesn't install/work on this hardware.

**Pattern D — the big one, PPO weight-sync fix** (`distributed_util.py`, `ppo_actor.py`,
`vllm_worker_wrap.py`, `deepspeed_utils.py`): see section 4, this is the most important fix and
has its own detailed report.

**Full line-by-line detail:** `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/CODE_REVIEW_all_files_changed.md`

**New test files added** (not modifications): `tests/test_advantage_estimators.py`,
`tests/test_loss_aggregation_xpu.py`, `tests/test_xccl_backend.py`. Full suite: **36/36 passing**
as of the last run (`cd OpenRLHF && venv-tf/bin/python -m pytest tests/ -q`).

---

## 3. Milestones proven working (all with real evidence, not just "ran without crashing")

| Method | Status | Evidence |
|---|---|---|
| SFT | ✅ | Real decreasing loss, checkpoint written |
| Ray | ✅ | Confirmed working (with a GPU-count auto-detect caveat — Ray doesn't see Intel GPUs automatically, must pass `--num-gpus` explicitly, detected via `torch.xpu.device_count()`, never hardcoded) |
| vLLM (standalone) | ✅ | Real inference: "The capital of France is" → "Paris. It is the largest city in Europe and" |
| RM training | ✅ | Found + fixed a real bug (see Pattern B above) |
| DPO training | ✅ | No bugs found |
| PPO (DeepSpeed actor training itself) | ✅ | Real policy loss, real gradients |
| PPO + vLLM weight sync | ✅ **Fixed 2026-07-09** | See section 4 — was silently broken, now fixed and verified |
| PPO across 5/6 advantage estimators (group_norm/GRPO, rloo, dr_grpo, reinforce, reinforce_baseline) | ✅ | All ran cleanly; `gae` untested (needs 2nd GPU for a real critic, not available on this machine) |

---

## 4. THE BIG FIX: PPO weight sync was silently broken, now fixed (2026-07-08 → 2026-07-09)

**Full detail in two reports:**
- `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/CORRECTION_weight_sync_not_real.md` (the bug + root-cause analysis)
- `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/WEIGHT_SYNC_FIX_implemented_and_verified.md` (the fix + verification)

**Summary:** PPO training looked like it was working (clean logs, real-looking metrics), but the
updated actor weights were **silently never reaching vLLM's rollout engine** — confirmed by
instrumenting the code to compare tensor checksums before/after the "broadcast" call across a
real training run: 0% of weight updates were actually transferred. Root cause: OpenRLHF's
`stateless_init_process_group()` always used NVIDIA's NCCL library, which doesn't exist on
Intel XPU, and silently no-op'd instead of erroring.

**Fix, in 2 stages:**
1. First fix attempt: route through a different fallback (`StatelessProcessGroup`'s own
   broadcast). Verified CORRECT (real transfers happening) but found to be **~540-700x slower
   than necessary** under GPU colocation (809 seconds for one weight-sync round on a tiny
   0.5B model — traced to a pickle-based, key-value-store transfer mechanism not meant for
   bulk data).
2. Final fix: switched to a real `gloo` process group (a genuine collective-communication
   backend). Same correctness (verified again with the checksum technique), but **~1.1-1.4
   seconds** per weight-sync round instead of 809 — confirmed on the same hardware, same
   workload, only the communicator implementation changed.

**This fix is what unblocked today's GRPO/GSM8K attempt** — before this fix, PPO/GRPO training
was running "successfully" but never actually improving the policy that generates rollouts.

---

## 5. Today's GRPO + GSM8K benchmark attempt — what's done, what's NOT done

**Goal:** run a real GRPO training run on GSM8K (grade-school math word problems, a standard
public benchmark) with Qwen2.5-1.5B-Instruct, compare pass@1 accuracy before vs. after training
to show a genuine, measurable lift — the kind of result that's comparable to published
NVIDIA-GPU benchmark numbers.

### ✅ DONE and solid:

1. **Baseline eval — complete, result saved.**
   - Model: `Qwen/Qwen2.5-1.5B-Instruct` (pretrained, no fine-tuning)
   - Dataset: 200 held-out GSM8K test questions (never used for training)
   - **Result: 66.0% pass@1 (132/200 correct)** — reasonably close to this model's published
     GSM8K numbers (~68-73% with different prompting), the gap is expected since we use a
     simple `\boxed{}`-only prompt with no few-shot examples.
   - Full result log: `/tmp/gsm8k_eval_results_baseline.json` (may or may not survive reboot,
     it's in `/tmp` — re-run is fast (~2 min) if lost, see section 8 for the exact command)
   - Eval log: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/logs/gsm8k_baseline_eval.log`

2. **Data preparation — complete, files persisted on disk (survive reboot).**
   - `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/data/gsm8k_train_prompts.jsonl` — 400 GSM8K training-split prompts, reformatted with a `\boxed{}` instruction suffix, `prompt`/`label` columns (OpenRLHF's expected format for `--reward.remote_url`-based custom reward functions).
   - `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/data/gsm8k_eval.jsonl` — 200 held-out GSM8K test-split prompts, same format. **This is the eval set — never train on this.**
   - Generation script (rerunnable, deterministic given the same GSM8K version): `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/scripts/prepare_gsm8k.py`

3. **Eval script — complete, reusable for both baseline and post-training checkpoints.**
   - `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/scripts/eval_gsm8k.py`
   - Usage: `venv-tf/bin/python eval_gsm8k.py <model_path> <eval_jsonl_path> --label <name>`
   - Uses OpenRLHF's own `extract_boxed_answer`/`grade_answer` utilities (same grader the
     reward function uses during training) so eval and training reward are consistent.

4. **Reward function — reused, not written from scratch.** OpenRLHF already ships
   `examples/python/math_reward_func.py` — extracts `\boxed{}` answer, grades via
   sympy/mathd-based comparison, returns 1.0/0.0. Used directly via `--reward.remote_url`.

### ❌ NOT DONE — the actual GRPO training run never completed successfully:

**6 attempts, in order, each hitting a different real issue:**

| # | Config | Result |
|---|---|---|
| 1 | `--ds.adam_offload` (mistakenly combined) + `--ds.zero_stage 0` | Crashed: DeepSpeed's CPUAdam assertion error (`adam_offload` needs a compatible ZeRO stage, this config combo is invalid) |
| 2 | Removed adam_offload, `--vllm.gpu_memory_utilization 0.25`, no sleep mode | Crashed: `ValueError: Free memory on device xpu:0 (0.0/30.3 GiB)` — DeepSpeed's actor+ref models alone filled the entire 32GB GPU before vLLM could even start (1.5B model is ~3x the 0.5B model used in all earlier successful tests) |
| 3 | Added `--vllm.enable_sleep --ds.enable_sleep` to let vLLM/DeepSpeed take turns using GPU memory | Crashed: `AttributeError: 'FP16_Optimizer' object has no attribute 'offload_states'` — `--ds.enable_sleep` needs a real ZeRO optimizer (`--ds.zero_stage >= 1`), we were still at `--ds.zero_stage 0` |
| 4 | `--ds.zero_stage 1` | Crashed: `AssertionError: Torch not compiled with CUDA enabled` — found and fixed a REAL, previously-unknown bug: `openrlhf/utils/deepspeed/deepspeed_utils.py`'s `offload_deepspeed_states()`/`reload_deepspeed_states()` (used by `--ds.enable_sleep`) had 4 hardcoded `torch.cuda.*` calls never hit until this exact feature combination was exercised. **Fixed** (now `torch.accelerator.*`, part of the 17 files in section 2). Pytest re-run: still 36/36. |
| 5 | Retried with the fix + `--ds.packing_samples` (an official OpenRLHF memory-saving flag) | Crashed: `ImportError: kernels is either not installed or uses an incompatible version (0.15.2 <= version < 0.16.0)` — our environment has `kernels==0.16.0` (required for the transformers-native flash-attention approach from earlier work), which conflicts with what `--ds.packing_samples` needs. **Not fixed** — dropped `--ds.packing_samples` instead of downgrading `kernels` (downgrading risked breaking the validated transformers-native setup). |
| 6 | Dropped `packing_samples`, added `--ds.adam_offload` (correctly this time, with `--ds.zero_stage 2`) + `--actor.gradient_checkpointing_enable`, per OpenRLHF's own official README OOM-remedy guidance | **Training actually started for real** — logged genuine rollouts and reward matches (e.g. `[Math Reward] Pred: 600, Gold: 600, Match: True`), vLLM's sleep mode successfully freed 7.21GB. Then hit a **system RAM OOM** (29.71GB/31.17GB) — not GPU — likely from `--ds.enable_sleep`'s CPU-side optimizer-state offload/reload cycling combined with 2 full 1.5B model copies' worth of CPU-resident optimizer state. |

**Then, investigating attempt 6's aftermath:** while cleaning up, a `PolicyModelActor` process
from attempt 6 became **permanently stuck** — 99% CPU for 45+ minutes, immune to `kill -9`,
holding `/dev/dri/renderD128` (the GPU device) open, causing `xpu-smi` itself to hang trying to
query the GPU. This is why a reboot is needed — see section 6.

### What we learned that should inform the NEXT attempt (don't repeat the same mistakes):

- **OpenRLHF's own official OOM-remedy order** (from their README, fetched and confirmed
  directly, this is the authoritative source, not guesswork):
  1. `--ds.packing_samples` + `--actor.gradient_checkpointing_enable`
  2. Reduce batch sizes
  3. Step down `--vllm.gpu_memory_utilization` (0.6 → 0.5 → 0.4)
  4. `--ds.adam_offload` + raise `--ds.zero_stage` (2 → 3)
  5. Remove `--train.colocate_*` (not viable for us — only 1 GPU on this machine)
- `--ds.packing_samples` is currently **incompatible with our environment** (`kernels==0.16.0`
  conflict) — either find a version of `kernels` that satisfies BOTH the transformers-native
  flash-attention requirement AND `packing_samples`'s requirement, or accept it stays disabled.
- `--ds.enable_sleep` + `--ds.adam_offload` together may be the specific combination that
  caused the system-RAM OOM in attempt 6 — the DeepSpeed code has an explicit early-return
  (`if adam_offload: return`) inside `offload_deepspeed_states()` suggesting they're meant to
  be **alternatives**, not necessarily combined. **Next attempt: try `adam_offload` WITHOUT
  `enable_sleep`** (accept slower/no GPU-memory-swapping between vLLM and DeepSpeed, rely on
  `adam_offload` alone to reduce GPU memory pressure), or vice versa — don't combine both again
  without understanding why attempt 6 OOM'd first.
- **Consider just using the smaller Qwen2.5-0.5B-Instruct model instead of 1.5B** — this model
  size was proven completely reliable across ~30 runs earlier in this session (SFT, RM, DPO,
  PPO×6 estimators, all colocated on this same 1 GPU) with zero memory issues at all. The 1.5B
  attempt has hit 6 different real errors (one being a genuine new bug we fixed) and finally a
  driver hang. This machine's 32GB GPU / 31GB system RAM may simply be undersized for
  comfortably colocating 2× 1.5B-parameter model copies + vLLM + all optimizer state, given
  everything else (Ray, VSCode, editor tooling) also competing for the same 31GB of RAM.

---

## 6. The stuck-GPU-process incident (why we're rebooting)

**Timeline:**
1. Attempt 6 (above) OOM'd on system RAM after real training had started.
2. Ray's own OOM killer killed one worker (`PPOTrainer`), but `PolicyModelActor` (a separate
   process, PID 235844 as of this writing) did not exit — it stayed running.
3. Attempted `kill -9` on 235844 repeatedly — process survived every attempt (confirmed
   `kill -9` returns exit code 0 / success, but the process remains in `ps` afterward).
4. Confirmed the process is genuinely running (state `R`, not `D`/blocked), burning ~99% CPU
   continuously, for over 45 minutes with zero progress in its log output.
5. Confirmed it holds `/dev/dri/renderD128` (the Intel GPU render device) open via
   `/proc/235844/fd/`.
6. Confirmed `xpu-smi discovery -d 0` (a completely separate, normally-instant diagnostic
   command) also hangs indefinitely while this process is alive — strong evidence the GPU
   driver itself is in a stuck/contended state, not just this one process being slow.

**Best-guess root cause (not fully confirmed, no root-cause debugging tools were run against
the kernel driver itself):** something in the interaction between DeepSpeed's `CPUAdam`
optimizer-state initialization (triggered by `--ds.adam_offload`) and the Intel XPU driver
stack, specifically for a 1.5B-parameter model under `--train.colocate_all`, caused a
driver-level operation to never return. This exact combination of flags had never been
exercised before this session (all ~30 earlier successful runs used the smaller 0.5B model and
never combined `adam_offload` + `colocate_all` + a model this size).

**If this recurs after reboot with the same flag combination:** do not retry
`--ds.adam_offload` on the 1.5B model under `--train.colocate_all` without further
investigation. Consider filing this as a genuine Intel XPU driver bug report if it's
reproducible, since a userspace process being unkillable via SIGKILL while holding a device
open is a kernel/driver-level issue, not an application bug.

---

## 7. IMPORTANT: uncommitted code changes — do not lose these

All 17 code fixes in `~/madhu/work2_with_transformers_native/OpenRLHF/` are **uncommitted**
(confirmed via `git status`/`git diff` — they show as modified files, not committed). A reboot
does NOT affect files on disk, so they will survive a restart fine. **But be careful not to run
any destructive git command** (`git checkout .`, `git reset --hard`, `git stash` followed by
forgetting to pop it, etc.) in that directory after reboot, or all of today's and yesterday's
fixes will be silently discarded. If in doubt, run `git diff --stat` first to confirm the 17
files still show as modified before doing anything else in that repo.

**Recommended next action once resumed:** consider committing these changes (with the user's
explicit go-ahead first, per this session's git-safety norms) so they're safe from any future
accidental `git` mistake. Not done yet as of this writing — needs an explicit ask.

---

## 8. Exact commands to resume from clean (after reboot + GPU verified working)

```bash
# 1. Confirm environment is healthy
xpu-smi discovery -d 0          # should return promptly
ps aux | grep -iE "ray::|EngineCore"   # should be empty
cd ~/madhu/work2_with_transformers_native/OpenRLHF && git diff --stat   # should show 17 files

# 2. Re-run pytest to confirm nothing regressed
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$(pwd)/venv-tf/bin:$PATH"
venv-tf/bin/python -m pytest tests/ -q     # expect 36/36

# 3. Re-verify baseline eval still holds (fast, ~2 min, optional if /tmp result survived)
venv-tf/bin/python ~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/scripts/eval_gsm8k.py \
  Qwen/Qwen2.5-1.5B-Instruct \
  ~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/data/gsm8k_eval.jsonl \
  --label baseline_recheck --gpu_memory_utilization 0.5

# 4. Start Ray fresh (always detect GPU count, never hardcode)
NUM_GPUS=$(venv-tf/bin/python -c "import torch; print(torch.xpu.device_count())")
ONEAPI_DEVICE_SELECTOR=level_zero:0 venv-tf/bin/ray start --head --num-gpus="$NUM_GPUS"

# 5. Decide: retry 1.5B with adjusted flags (see section 5's lessons-learned), or switch to
#    0.5B for reliability. Either way, apply OOM remedies in OpenRLHF's official order (section 5).
```

## 9. Where every report/file referenced above actually lives

All under `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/`:
- `reports/CODE_REVIEW_all_files_changed.md` — every code fix, line by line
- `reports/CORRECTION_weight_sync_not_real.md` — the weight-sync bug, root cause
- `reports/WEIGHT_SYNC_FIX_implemented_and_verified.md` — the fix, both iterations, verification
- `reports/rlhf_methods_coverage.md`, `ray_and_ppo_findings.md`, `vllm_xpu_build_and_ppo_success.md`, `venv-tf_setup_and_code_changes.md` — earlier milestone reports (SFT/RM/DPO/Ray/vLLM build)
- `data/gsm8k_train_prompts.jsonl`, `data/gsm8k_eval.jsonl` — prepared datasets (done, reusable)
- `scripts/prepare_gsm8k.py`, `scripts/eval_gsm8k.py` — reusable scripts
- `logs/gsm8k_baseline_eval.log`, `logs/gsm8k_grpo_training.log` through `training6.log` — full logs of every attempt described in section 5
- **This file:** `reports/RESUME_AFTER_REBOOT.md`
