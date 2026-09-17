# Single-XPU OpenRLHF: code changes + full PPO topology matrix

**Scope:** single XPU only. Do NOT port anything multi-GPU-specific (XCCL weight-sync backend,
the weight-freshness diagnostic probe, `--vllm.sync_backend xccl`). Deferred to a later pass.

**Starting point:** `https://github.com/Madhustat/OpenRLHF/tree/xpu/experimental-e2e-baseline`
(commit `ebe82c50`). Verified by diffing that branch against a validated fix branch: it ALREADY
has device-API portability (`torch.cuda`→`torch.accelerator`), the `flash_attn` optional
fallback, vLLM device pinning, gloo weight-sync, and 7 test files. **Do not redo those.**

---

# PART 1 — Environment (torch 2.13 stack)

```bash
# PyTorch — torchvision/torchaudio are NOT needed and NOT installed
pip install torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
pip install triton-xpu==3.7.2

# vLLM — source build; no PyPI wheel exists for this pairing
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 6e448d0ea                       # == v0.27.2.dev0+g6e448d0ea
VLLM_TARGET_DEVICE=xpu pip install -e . --no-build-isolation
pip install "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.12/vllm_xpu_kernels-0.1.12-cp38-abi3-manylinux_2_28_x86_64.whl"

# Remaining deps
pip install ray==2.55.0 deepspeed==0.19.1 transformers==5.7.0
```

Verified working stack: Python 3.13.12, torch 2.13.0+xpu, vLLM 0.27.2.dev0+g6e448d0ea,
vllm-xpu-kernels 0.1.12, DeepSpeed 0.19.1, Ray 2.55.0, transformers 5.7.0, oneCCL 2022.0.0,
Intel runtime 2026.0.0, kernel 7.0.0-31-generic (`xe`), libze-intel-gpu1 26.22.38646.6.

## Three environment facts that are load-bearing

```bash
# 1. venv lib/ MUST precede any base conda/miniforge lib/, or torch 2.13's libsycl.so.9
#    fails at import with "undefined symbol: urDeviceWaitExp"
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"

# 2. No icpx compiler installed -> DeepSpeed cannot JIT-build FusedAdam -> force torch AdamW.
#    This is WHY the Part-2 fixes are needed at all.
export OPENRLHF_DS_TORCH_ADAM=1

# 3. dpctl deliberately NOT installed (conflicts with this runtime) -> Ray cannot discover XPUs
#    and reports GPU=0, so every placement group hangs. Declare the count explicitly:
ray start --head --num-gpus 1

# Plus, to pin to one device when 2 are physically present:
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
```

---

# PART 2 — Code changes (3 files)

## Change 1a — `openrlhf/utils/deepspeed/deepspeed_utils.py`: ZeRO-3 native crash

In `get_train_ds_config()`, find:

```python
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
    if stage == 3:
        zero_opt_dict["reduce_scatter"] = True
```

Replace with:

```python
    if overlap_comm:
        zero_opt_dict["overlap_comm"] = True
        zero_opt_dict["contiguous_gradients"] = True
    elif stage == 3:
        # DeepSpeed's own ZeRO config resolves overlap_comm=None to
        # `self.stage == ZeroStageEnum.weights` (True for stage 3) whenever the key is absent --
        # so omitting it here to mean "False" silently becomes "True" for stage 3, the opposite
        # of this function's own overlap_comm=False default. Causes a native SIGSEGV in oneCCL's
        # progress thread on Intel XPU under ATL/OFI. Must be written explicitly.
        # CONFIRMED RANK-COUNT-INDEPENDENT: crashes with a single actor (world_size=1) too, so
        # this is NOT a multi-GPU-only concern.
        zero_opt_dict["overlap_comm"] = False
    if stage == 3:
        zero_opt_dict["reduce_scatter"] = True
```

## Change 1b — same file: DeepSpeed-sleep crash with non-FusedAdam optimizer

Add above `offload_deepspeed_states`:

```python
def _optimizer_supports_state_offload(model):
    """stage3.py's offload_states() hard-asserts
    `self.optimizer.__class__ == deepspeed.ops.adam.fused_adam.FusedAdam` before touching either
    OffloadStateTypeEnum.optim_states or .hp_params. The other three categories (lp_params,
    lp_grads, contiguous_grad_buffer) never reference self.optimizer at all -- they only move
    DeepSpeed's own ZeRO-3 partition buffers -- so they work with any optimizer.
    """
    inner_optimizer = getattr(model.optimizer, "optimizer", None)
    return inner_optimizer is not None and inner_optimizer.__class__ is deepspeed.ops.adam.fused_adam.FusedAdam
```

In `offload_deepspeed_states()`, find:

```python
    offload_state_types = [
        OffloadStateTypeEnum.optim_states,
        OffloadStateTypeEnum.contiguous_grad_buffer,
        OffloadStateTypeEnum.hp_params,
    ]
```

Replace with:

```python
    fused_adam_capable = _optimizer_supports_state_offload(model)

    offload_state_types = [
        OffloadStateTypeEnum.contiguous_grad_buffer,
    ]
    if fused_adam_capable:
        offload_state_types += [
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.hp_params,
        ]

    if not fused_adam_capable and not getattr(model, "_openrlhf_partial_sleep_logged", False):
        inner = getattr(model.optimizer, "optimizer", None)
        print(
            f"[deepspeed sleep] partial offload mode: optimizer={type(inner).__name__} does not "
            f"support FusedAdam-only state offload -- retaining optim_states, hp_params on-device; "
            f"offloading {[t.name for t in offload_state_types]}",
            flush=True,
        )
        model._openrlhf_partial_sleep_logged = True
```

Leave the rest of the function unchanged (the `lp_grads` version-check block and the final
`model.optimizer.offload_states(...)` call).

**IMPORTANT SCOPE NOTE:** this assert lives in `stage3.py` only. ZeRO stages 1 and 2 use
`stage_1_and_2.py`, which has NO such assert and already uses a generic optimizer-state mover.
So **this fix is needed ONLY for ZeRO-3**; stages 1/2 DS-sleep works without it on DeepSpeed
0.19.1. (Verified by reading both source files.)

## Change 2 — `openrlhf/utils/deepspeed/deepspeed.py`: torch-AdamW escape hatch

In the AdamW optimizer-config branch (`"type": "AdamW"`), add the last line of `params`:

```python
            optim_dict = {
                "type": "AdamW",
                "params": {
                    "lr": adam["lr"],
                    "betas": list(adam["betas"]),
                    "eps": adam["eps"],
                    "weight_decay": adam["weight_decay"],
                    # Opt-in escape hatch: force DeepSpeed to use torch's native AdamW instead
                    # of its fused XPU op. Without icpx, FusedAdam/CPUAdam cannot be JIT-built.
                    # Defaults OFF so any working CUDA path stays byte-for-byte unchanged.
                    **({"torch_adam": True} if os.environ.get("OPENRLHF_DS_TORCH_ADAM", "0") == "1" else {}),
                },
            }
```

(`import os` at the top if not already present.)

## Change 3 — `openrlhf/trainer/ray/ppo_actor.py`: hardcoded `"cuda"` breaks EMA

Two occurrences of:

```python
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
```

Replace BOTH with:

```python
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, torch.accelerator.current_accelerator().type)
```

Breaks `--train.enable_ema` on any non-CUDA accelerator. Pure portability bug, unrelated to GPU count.

## Do NOT port (out of scope for single GPU)

- **XCCL weight-sync** (`distributed_util.py`'s `_XcclBroadcastCommunicator`,
  `_init_xccl_process_group`, backend-resolver rewrite). Only relevant with 2+ physical devices;
  the baseline's existing resolver already correctly returns `gloo` for one XPU. Skip the file.
- **Weight-freshness probe** (`ppo_actor.py`'s `OPENRLHF_WEIGHT_PROBE` block and its
  `vllm_worker_wrap.py` half). Guards a deadlock only possible at actor `world_size > 1` —
  structurally unreachable on one device.
- **`--vllm.sync_backend xccl`** CLI choice. Leave `nccl`/`gloo` as-is.

---

# PART 3 — Full single-GPU PPO topology matrix

## Why the axes are what they are (derived from the CLI, not invented)

- **Colocation is NOT an axis.** `--train.colocate_all` is mandatory on 1 GPU: without it the
  actor, critic and vLLM engine each request an exclusive device (3 GPUs). Every valid
  single-GPU topology sets it.
- **Async mode is entirely out of scope.** `train_ppo_ray.py` asserts `colocate_all` and
  `async_enable` are mutually exclusive, and async requires vLLM on its own device.
- **Critic presence is NOT independently settable.** It is derived from the advantage estimator:
  `gae` ⇒ critic; anything else ⇒ `critic.model_name_or_path` is forced to `None`.
  Non-`gae` group estimators additionally require `n_samples_per_prompt > 1`.
- **`adam_offload` is a real axis and interacts with sleep.**
  `offload_deepspeed_states()` returns immediately when `adam_offload` is on, so
  **DS-sleep becomes a silent no-op** — those cells run clean but prove nothing about sleep.

So the axes are: **critic (2) × adam_offload (2) × sleep (4) × ZeRO stage (4) = 64 cells.**

## Legend

| Mark | Meaning |
|---|---|
| `VALIDATED` | Already exercised by the existing single-GPU E2E suite |
| `TO TEST` | Expected to pass; not yet run in the full Ray+vLLM stack |
| `FIX REQUIRED` | Needs Part-2 changes; without them this cell crashes |
| `SLEEP NO-OP` | Runs clean, but DS-sleep is silently disabled by `adam_offload` |
| `UNSUPPORTED` | Architecturally impossible — do not chase |

## Group A — PPO / GAE (critic present), `adam_offload = Off`

| # | Sleep | Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| A1 | Both Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| A2 | vLLM On / DS Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| A3 | vLLM Off / DS On | UNSUPPORTED | TO TEST | TO TEST | FIX REQUIRED (1a+1b) |
| A4 | Both On | UNSUPPORTED | TO TEST | TO TEST | FIX REQUIRED (1a+1b) |

## Group B — PPO / GAE (critic present), `adam_offload = On`

| # | Sleep | Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| B1 | Both Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| B2 | vLLM On / DS Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| B3 | vLLM Off / DS On | SLEEP NO-OP | SLEEP NO-OP | SLEEP NO-OP | SLEEP NO-OP + FIX (1a) |
| B4 | Both On | SLEEP NO-OP | SLEEP NO-OP | SLEEP NO-OP | SLEEP NO-OP + FIX (1a) |

## Group C — GRPO family (no critic), `adam_offload = Off`

Estimators: `group_norm` (GRPO), `reinforce_baseline`, `rloo`, `dr_grpo`. Require
`n_samples_per_prompt > 1`.

| # | Sleep | Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| C1 | Both Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| C2 | vLLM On / DS Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| C3 | vLLM Off / DS On | UNSUPPORTED | TO TEST | TO TEST | FIX REQUIRED (1a+1b) |
| C4 | Both On | UNSUPPORTED | TO TEST | TO TEST | FIX REQUIRED (1a+1b) |

## Group D — GRPO family (no critic), `adam_offload = On`

| # | Sleep | Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|
| D1 | Both Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| D2 | vLLM On / DS Off | TO TEST | TO TEST | TO TEST | FIX REQUIRED (1a) |
| D3 | vLLM Off / DS On | SLEEP NO-OP | SLEEP NO-OP | **VALIDATED** | SLEEP NO-OP + FIX (1a) |
| D4 | Both On | SLEEP NO-OP | SLEEP NO-OP | **VALIDATED** | SLEEP NO-OP + FIX (1a) |

## Why the `UNSUPPORTED` and `VALIDATED` marks

- **Stage 0 + DS-sleep, `adam_offload=Off`** (A3, A4, C3, C4): DeepSpeed's plain
  `FP16_UnfusedOptimizer` has no `offload_states()` method at all — raises
  `'FP16_UnfusedOptimizer' object has no attribute 'offload_states'`. Architectural, not fixable
  here. Note it becomes `SLEEP NO-OP` rather than a crash when `adam_offload=On` (B3/B4/D3/D4),
  because the early return fires before the missing method is reached.
- **D3/D4 ZeRO-2 = VALIDATED**: the existing `tests/test_e2e_suite_singlegpu.sh` (19 tests)
  sweeps exactly this cell — `--ds.zero_stage 2 --ds.adam_offload` with `--train.colocate_all
  --vllm.enable_sleep --ds.enable_sleep` — across GRPO, REINFORCE, REINFORCE++, RLOO, DR-GRPO,
  LoRA, KL-penalty and reward-normalisation variants.
- **Everything else is untested on a single GPU.** In particular the existing suite never
  exercises ZeRO-3 at all, and never exercises DS-sleep with `adam_offload=Off` (i.e. real
  optimizer-state offload). Those are the genuinely new cells.

## Highest-value cells to run first

1. **A4 / C4, ZeRO-3** — ZeRO-3 + both sleeps + `adam_offload=Off`. Exercises BOTH Part-2 fixes
   and is the only configuration where DS-sleep actually offloads optimizer state. Never run in
   the full single-GPU Ray+vLLM+Critic stack.
2. **A1, ZeRO-3** — plain ZeRO-3 PPO with a critic, no sleep. Isolates fix 1a from fix 1b.
3. **A3/C3, ZeRO-1 and ZeRO-2** — DS-sleep with `adam_offload=Off` at stages 1/2. Should pass
   without any Part-2 fix (different DeepSpeed code path); confirms that claim.
4. Then sweep the remainder.

## Run recipe

```bash
export OPENRLHF_DS_TORCH_ADAM=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
ray start --head --num-gpus 1

python -m openrlhf.cli.train_ppo_ray \
  --actor.num_gpus_per_node 1 \
  --critic.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --train.colocate_all \
  --ds.zero_stage 3 \
  --ds.enable_sleep --vllm.enable_sleep \
  --algo.advantage.estimator gae \
  --vllm.gpu_memory_utilization 0.22 \
  --ds.attn_implementation sdpa \
  <model / dataset / batch args>

# Between every run: stop ray, confirm no stray raylet/vLLM procs, confirm XPU memory released.
ray stop
```

For the GRPO-family groups (C/D), swap `--algo.advantage.estimator gae` for e.g.
`group_norm` and set `--rollout.n_samples_per_prompt 4` (must be > 1).
For groups B/D add `--ds.adam_offload`.

## Pass criteria — do not accept "it didn't crash"

For each cell record: target steps reached, exit code 0, no `AssertionError`, no SIGSEGV, and
that actor→vLLM weight sync actually happened (weight checksums match after a broadcast, or at
minimum that the rollout reflects the updated policy). A run that completes without ever
successfully syncing weights is a false pass.

---

# PART 4 — Validation before the sweep

1. **Regression tests (no GPU, ~5 s).** Port `tests/test_ppo_zero3_fixes_suite.py` from the
   `fix/ppo-zero3-deepspeed-sleep` branch. Only its `overlap_comm` (3 tests) and
   `FusedAdam`-capability (3 tests) sections apply here; its 3 weight-probe tests cover code
   deliberately not ported — drop that section or expect them to fail on the missing symbol.

2. **Existing suites still pass.** `pytest tests/ -q` and
   `bash tests/test_e2e_suite_singlegpu.sh` — confirm the Part-2 changes broke nothing that
   already worked.

3. Then work the matrix, highest-value cells first.

---

# Known limitation to flag, not solve

With fix 1b, `--ds.enable_sleep` on ZeRO-3 recovers materially less memory than DeepSpeed's
`FusedAdam` path would (roughly a quarter to a third of what is achievable, measured). The cause
is a restriction in DeepSpeed itself — it permits full optimizer-state offload only for
`FusedAdam`, even though the underlying mechanism is bitwise-correct with plain `torch.optim.AdamW`
(verified experimentally at both 1 and 2 ranks). The proper fix belongs upstream in DeepSpeed. If
single-GPU memory headroom is tight enough that this matters, escalate rather than work around it
locally.
