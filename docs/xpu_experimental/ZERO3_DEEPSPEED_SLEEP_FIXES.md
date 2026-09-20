# ZeRO-3 and DeepSpeed-sleep fixes for Ray PPO on Intel XPU

## Purpose of this branch

This branch takes the gloo-based vLLM weight-synchronization work (`openrlfh_exp_multi`) and
adds the minimum set of source changes needed to make **Ray PPO training pass on every
architecturally-valid topology at ZeRO stage 3**, including with DeepSpeed sleep
(`--ds.enable_sleep`) and vLLM sleep (`--vllm.enable_sleep`) enabled.

Before these changes, ZeRO-3 failed on this stack in two independent ways: a native crash on
every ZeRO-3 run, and a hard assertion failure on every ZeRO-3 + DeepSpeed-sleep run. Both are
bugs in OpenRLHF's own logic, not in Intel/XPU-specific code paths — they were simply exposed
by testing here.

## Validated on torch 2.13

Everything here was validated on **`torch 2.13.0+xpu`** — not 2.12. The full pinned stack
(vLLM `0.27.2.dev0+g6e448d0ea` source build, DeepSpeed 0.19.1, Ray 2.55.0, transformers 5.7.0,
oneCCL 2022.0.0, Python 3.13.12) and the exact install procedure are in
[INSTALL_XPU.md](INSTALL_XPU.md), which was updated to this stack alongside these fixes.

Hardware: 2x Intel Arc Pro B70 (Battlemage, PCIe, no XeLink), kernel `7.0.0-31-generic` (`xe`).

Two environment properties are load-bearing for reproducing these results:

- **`icpx` is not installed**, so DeepSpeed cannot JIT-build `FusedAdam` and
  `OPENRLHF_DS_TORCH_ADAM=1` is required. This is what makes Issue 2 below reachable at all.
- **`dpctl` is not installed** (it conflicts with this runtime), so Ray cannot discover XPUs
  and the head must be started with `ray start --head --num-gpus 2`.

## What changed, and which file fixes which issue

Three files change in total. Only the first is relevant to normal production training.

| Issue | File changed | Production-relevant? |
|---|---|---|
| ZeRO-3 native SIGSEGV (`overlap_comm`) | `openrlhf/utils/deepspeed/deepspeed_utils.py` | **Yes**, all backends |
| DeepSpeed sleep crash with non-FusedAdam optimizer | `openrlhf/utils/deepspeed/deepspeed_utils.py` | **Yes**, when `icpx` is unavailable |
| ZeRO-3 weight-probe deadlock | `openrlhf/trainer/ray/ppo_actor.py` | No — diagnostic-only (see below) |
| Regression tests for all three | `tests/test_ppo_zero3_fixes_suite.py` | Test-only |

### Issue 1 — ZeRO-3 native SIGSEGV, fixed in `deepspeed_utils.py`

`get_train_ds_config()` only *wrote* the `overlap_comm` key into the ZeRO config dict when the
value was `True`; when `False` it omitted the key entirely. DeepSpeed's own ZeRO config
resolves an **absent** `overlap_comm` to `self.stage == ZeroStageEnum.weights`, which is `True`
for stage 3. So `--ds.overlap_comm`'s own CLI default of `False` never reached the live config:
**every ZeRO-3 run silently ran with `overlap_comm=True`**, the opposite of what the flag said.

On this hardware that async gradient-reduce/backward overlap races with oneCCL's Level-Zero
event lifecycle under the ATL/OFI transport, producing a SIGSEGV in oneCCL's background
progress thread (`ccl_worker_func -> ... -> urEventGetInfo -> libze_intel_gpu.so.1`).

Isolated to a single variable: adding exactly one line (`"overlap_comm": False`) to an
otherwise-unmodified crashing reproducer — same transport, same launcher, nothing else changed
— took it from crashing every run to 3/3 clean runs, 300 combined steps.

Fix: write `overlap_comm=False` explicitly for stage 3 instead of omitting the key. Scoped to
stage 3 only; stages 1/2 keep the original omit-if-false behaviour, since DeepSpeed's
silent-`True` trap is specific to `ZeroStageEnum.weights`.

Note for review: because this makes the CLI's documented default actually take effect, it is a
**behaviour change on every backend**, not just XPU. Anyone previously (unknowingly) getting
`overlap_comm=True` on CUDA will now get `False` unless they pass `--ds.overlap_comm`.

### Issue 2 — Adam / DeepSpeed-sleep crash, fixed in `deepspeed_utils.py`

`--ds.enable_sleep` calls `offload_deepspeed_states()`, which unconditionally requested
`OffloadStateTypeEnum.optim_states` and `.hp_params`.
`DeepSpeedZeroOptimizer_Stage3.offload_states()` hard-asserts
`self.optimizer.__class__ == deepspeed.ops.adam.fused_adam.FusedAdam` before touching either of
those two categories (`deepspeed/runtime/zero/stage3.py:3285`) — both call into
FusedAdam-specific helpers. Result:

```
AssertionError: Offloading is supported only for DeepSpeed FusedAdam
```

on every ZeRO-3 + `--ds.enable_sleep` run, because `icpx` is not available here to JIT-build
`FusedAdam`, so OpenRLHF falls back to plain `torch.optim.AdamW`
(`OPENRLHF_DS_TORCH_ADAM=1`).

The other three offload categories (`lp_params`, `lp_grads`, `contiguous_grad_buffer`) never
reference `self.optimizer` at all — they only move DeepSpeed's own ZeRO-3 partition buffers, so
they work with any optimizer.

Fix: `_optimizer_supports_state_offload()` performs a capability check (is the optimizer
actually `FusedAdam`?) and `offload_deepspeed_states()` requests the two FusedAdam-only
categories only when that holds, falling back to the optimizer-independent categories
otherwise, with a one-time log line stating what is retained on-device. **No optimizer is
swapped and no compiler is required**; the `FusedAdam` path is byte-for-byte unchanged for
anyone using it, so this is additive for existing CUDA users.

Side effect: retaining `optim_states`/`hp_params` on-device means less memory churn during
offload/reload, which also resolved a vLLM memory-profiling assertion
(`Error in memory profiling. Initial free memory X, current free memory Y`) that previously
appeared only when DeepSpeed sleep and vLLM sleep were both enabled.

### Issue 3 — ZeRO-3 weight-probe deadlock, fixed in `ppo_actor.py` (diagnostic-only)

The `OPENRLHF_WEIGHT_PROBE` block gated its entire body — including a real
`deepspeed.zero.GatheredParameters` call — on `torch.distributed.get_rank() == 0`. Under ZeRO-3
with actor world size > 1, `GatheredParameters` issues a real collective matched by *call
order*, so rank 0 issuing probe-only collectives that no other rank issues permanently shifted
it out of phase, deadlocking the next `GatheredParameters` in the real broadcast loop. Confirmed
with `py-spy`: rank 0 stuck gathering one parameter while rank 1 was three parameters ahead in
the main sync loop.

Fix: every rank enters the identical `GatheredParameters` call in the same order; only the
checksum read and log line stay rank-0-only.

**This is not needed for stock production training** — `OPENRLHF_WEIGHT_PROBE` is off by
default, so the buggy path never executes in a normal run. It was required for this
investigation's own methodology, which used the probe to verify actor/vLLM weight checksums
actually matched after each broadcast rather than merely asserting "did not crash".

## vLLM KV-cache / memory sizing — no code change

Several ZeRO-1/ZeRO-2 cases had previously been recorded as having a "vLLM KV-memory sizing
issue". After the two `deepspeed_utils.py` fixes landed, all of them were re-run **unmodified**
and passed with healthy KV-cache headroom at the existing `--vllm.gpu_memory_utilization 0.22`:

| Case | Result | Available KV cache memory |
|---|---|---|
| X18-Z1 | PASS 5/5 | 5.24 GiB |
| X19-Z1 | PASS 5/5 | 6.11 / 5.18 GiB (2 engines) |
| X19-Z2 | PASS 5/5 | 5.41 / 5.24 GiB (2 engines) |
| X23-Z1 | PASS 5/5 | 5.69 GiB |
| X23-Z2 | PASS 5/5 | 5.69 GiB |

Those entries were stale, recorded before the fixes above. Memory sizing is controlled entirely
by the runtime flag `--vllm.gpu_memory_utilization`; **no source change was needed or made for
it.**

## Validation

- ZeRO-3, DeepSpeed sleep **off** — 9/9 valid topologies pass 5/5 steps
  (X4, X10, X12, X14, X16, X18, X20, X22, X24); X12 repeated 3/3 clean runs.
- ZeRO-3, DeepSpeed sleep **on** — 9/9 valid topologies pass 5/5 steps
  (X9, X11, X13, X15, X17, X19, X21, X23); X9 repeated 3x, all pass.
- Every pass includes real Gloo actor-to-vLLM weight sync with matching actor/vLLM parameter
  checksums and a real rollout on the updated policy — not merely absence of a crash.
- Excluded by design, never run: X5-X8 (critic world size > actor world size is unschedulable
  on a 2-GPU box) and X1/X2 (async mode conflicts with vLLM sleep).

## Running the regression tests

All three fixes are covered by one file, runnable with a single command. No GPU/XPU, no live
Ray cluster, and no DeepSpeed accelerator runtime required — the tests exercise the
config-building logic and the trainer's source shape, which is what the fixes changed.

```bash
pytest tests/test_ppo_zero3_fixes_suite.py -v
```

or:

```bash
python tests/test_ppo_zero3_fixes_suite.py
```

Expected: 9 passed.
