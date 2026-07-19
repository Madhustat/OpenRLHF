# CORRECTION — PPO weight sync to vLLM never actually transferred data on Intel XPU

> **✅ RESOLVED (2026-07-09): see `WEIGHT_SYNC_FIX_implemented_and_verified.md`.**
> The fix from the "Recommended fix" section below has been implemented, verified correct
> (live checksum instrumentation, 572/572 real transfers) AND made fast (switched from the
> pickle-based `StatelessProcessGroup` to a real `gloo` ProcessGroup after finding the former
> was ~540-700x slower than necessary under GPU colocation — 809s → ~1.3s per weight-sync
> round). PPO+vLLM on Intel XPU is now genuinely working, not just training-metrics-look-real.

Environment: `~/madhu/work2_with_transformers_native/OpenRLHF/venv-tf`
Hardware: Intel Arc Pro B70 (Battlemage), Ubuntu 24.04.3
Found: 2026-07-08 (evening), during a comparison against a colleague's verl Dockerfile

**This corrects a claim in `vllm_xpu_build_and_ppo_success.md` and `rlhf_methods_coverage.md`.**
Every PPO run reported there as a full success ("weight sync confirmed across all layers")
did NOT actually transfer weight data to vLLM. The DeepSpeed-side actor training was real;
the rollout generation vLLM used was running on stale, never-updated weights the whole time.

---

## What we previously claimed

Both prior reports state weight sync was "confirmed" based on log lines like:
```
(EngineCore pid=140707) update weight: model.layers.19.self_attn.q_proj.weight, ...
```
appearing for every transformer layer, for every PPO advantage estimator tested
(`group_norm`, `rloo`, `dr_grpo`, `reinforce`, `reinforce_baseline`).

**This was the wrong signal to trust.** That print statement fires unconditionally, before
the actual broadcast call — it only proves `update_weight()` was invoked once per parameter,
not that any bytes moved.

## What's actually happening

`openrlhf/utils/distributed_util.py`'s `stateless_init_process_group()` always constructs a
`vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`, regardless of the
`--vllm.sync_backend` CLI flag's value (`gloo` or `nccl` — the `backend` string is passed
around and even forwarded to `engine.init_process_group.remote(..., backend=backend)`, but
`stateless_init_process_group()` itself never reads it; it's dead as far as this code path is
concerned).

`PyNcclCommunicator.__init__` requires the NCCL shared library. vLLM's own
`find_nccl_library()` (`vllm/distributed/device_communicators/pynccl_wrapper.py`):
```python
if torch.version.cuda is not None:
    so_file = "libnccl.so.2"
elif torch.version.hip is not None:
    so_file = "librccl.so.1"
else:
    raise ValueError("NCCL only supports CUDA and ROCm backends.")
```
On Intel XPU, `torch.version.cuda` and `torch.version.hip` are both `None` — this always
raises. `PyNcclCommunicator.__init__` catches this with a bare `except Exception:` and sets
`self.disabled = True` instead of propagating the error:
```python
try:
    self.nccl = NCCLLibrary(library_path)
except Exception:
    self.available = False
    self.disabled = True
    return
```
`PyNcclCommunicator.broadcast()` then no-ops when disabled:
```python
def broadcast(self, tensor, src, stream=None):
    if self.disabled:
        return
    ...
```
**No exception, no warning, no log line — the broadcast call simply returns immediately.**
This is why every "successful" run looked completely clean.

## Direct verification (not just static code reading)

Confirmed with two independent tests, both against the exact production code path:

1. **Standalone 2-process repro**, calling OpenRLHF's real `stateless_init_process_group()`
   helper with `world_size=2` (matching our actual PPO configs):
   ```
   [rank 0] available=False disabled=True
   [rank 1] available=False disabled=True
   ```

2. **Live instrumented PPO run** (temporary debug code added to `vllm_worker_wrap.py`'s
   `update_weight()`, removed after this test — not left in the codebase): printed
   `pynccl.disabled` and compared each weight tensor's checksum before/after the broadcast
   call, for a real `group_norm` PPO run (same config as prior successful reports).
   **Result: `disabled=True` for all 561 weight-update calls across the full run (1,122 debug
   log lines total — 2 lines per call). 0 of them showed a real, non-NaN checksum change.**
   203 lines showed `changed=True`, but every one of those is a `nan`-before/`nan`-after
   artifact (`nan != nan` in Python) — verified by re-grepping the saved log directly: 203
   `changed=True` lines, 203 of which contain `nan`, 0 with any other value.

## Scope of the correction

Affects every PPO run in both prior reports:
- `vllm_xpu_build_and_ppo_success.md` — the original `group_norm` PPO+vLLM milestone run
- `rlhf_methods_coverage.md` — `rloo`, `dr_grpo`, `reinforce`, `reinforce_baseline` runs

**Does NOT affect:**
- SFT, RM, DPO training — none of these use vLLM or this code path at all
- The DeepSpeed-side actor training itself — policy/value loss, gradient updates, and
  optimizer steps were all real and correct; only the *copy* of updated weights into vLLM's
  rollout engine was silently skipped
- `update_weight_cuda_ipc()` — already known and documented as unfixed/deferred; unaffected
  by this finding since it isn't the path we were using
- The `torch.accelerator.*` fixes (device dispatch, empty_cache, etc.) — all still correct
- vLLM's own inference correctness (`LLM.generate()` on its own, without OpenRLHF's actor
  sync layer, produced the correct "Paris" output — that was tested completely independently
  of this weight-sync path)

## Revised interpretation of "PPO+vLLM smoke tests"

What we actually validated: OpenRLHF's DeepSpeed-based actor training loop runs correctly on
XPU, and vLLM can independently generate real completions on XPU. What we have NOT validated:
that a PPO training loop actually improves the policy vLLM uses for rollout generation — the
core loop that makes PPO/GRPO meaningfully different from just repeatedly sampling a frozen
model. The training metrics logged (loss, reward, kl) are still real numbers computed from
real generations — they're just computed against a policy that never received its updates
during the run.

## Root-cause analysis of fix candidates

Four candidates were tested with real 2-process scripts against XPU tensors (not just read
from source) before settling on a recommendation. All temporary test scripts were deleted
after use — none of this touched the actual OpenRLHF/vLLM codebase.

| # | Candidate | Verified result |
|---|---|---|
| 1 | Swap in vLLM's `XpuCommunicator` (`vllm/distributed/device_communicators/xpu_communicator.py`) — the class referenced but not investigated when this gap was first flagged in `vllm_xpu_build_and_ppo_success.md` | **Ruled out — confirmed by actually constructing it, not just reading signatures.** Passing our `StatelessProcessGroup` as `cpu_group`/`device_group` raises immediately: `TypeError: unhashable type: 'StatelessProcessGroup'`, from `DeviceCommunicatorBase.__init__`'s `_world.pg_map.get(cpu_group, None)` — it expects a real hashable `torch.distributed.ProcessGroup`, not vLLM's lightweight TCP-store-backed metadata object. Not a drop-in swap without deeper restructuring. |
| 2 | Real `dist.init_process_group(backend='gloo')` + move tensor to CPU + broadcast + copy back to XPU | Confirmed working with a real 2-process test (value `99.0` correctly arrived on the receiving rank). **Rejected as risky**: this creates a second **default** process group per Python process. DeepSpeed already owns the default group (`xccl` backend) in the actor process, and vLLM's engine process has its own internal groups — layering another default-group `init_process_group` call risks collision. |
| 3 | `stateless_init_torch_distributed_process_group(backend='gloo')` (vLLM's own non-default group constructor), then call `.broadcast()` directly on the returned `ProcessGroupGloo` object | Confirmed working with a real 2-process test (value `123.0` arrived correctly). vLLM's own docstring warns broadcast "cannot be used" on this — but that warning is about the module-level `torch.distributed.broadcast()` function (which resolves the group by identity against a global registry this stateless group isn't part of), not about calling `.broadcast()` as a method directly on the returned `pg` object, which works fine. Viable, but pulls in more of vLLM's internal API than candidate 4. |
| 4 | **Use `StatelessProcessGroup.broadcast()` directly — the object `stateless_init_process_group()` already constructs internally, just skip wrapping it in `PyNcclCommunicator`** | **Confirmed working, recommended fix.** Real 2-process test with a realistic bf16 tensor (shape `(4864, 896)`, matching an actual Qwen2.5-0.5B `mlp.up_proj` layer): correct value (`3.5`) arrived on the receiving rank in ~30-50ms. Minimal change — the object already exists in the code, we're currently discarding it in favor of a communicator that silently fails on XPU. |

### The catch with candidate 4 (and 2/3): return value, not in-place mutation

`PyNcclCommunicator.broadcast()` mutates its `tensor` argument in-place (real NCCL buffer
semantics), so both current call sites discard the return value:
```python
# openrlhf/trainer/ray/vllm_worker_wrap.py:46
self._model_update_group.broadcast(weight, src=0)   # return value discarded
# openrlhf/trainer/ray/ppo_actor.py:435
self._model_update_group.broadcast(param.data, src=0)   # return value discarded
```
`StatelessProcessGroup.broadcast()` does **not** mutate in-place — confirmed directly: the
receiving rank's original tensor is unchanged after the call; the received data only exists in
the **returned** tensor. Fixing this requires capturing the return value at both call sites,
e.g. `weight = self._model_update_group.broadcast(weight, src=0)` — not just swapping which
communicator class gets constructed.

### Recommended fix (not yet implemented)

In `openrlhf/utils/distributed_util.py`'s `stateless_init_process_group()`: detect XPU (or
more generally, "NCCL unavailable") and return the `StatelessProcessGroup` (`pg`) directly
instead of wrapping it in `PyNcclCommunicator`. Then update both call sites
(`ppo_actor.py:435`, `vllm_worker_wrap.py:46`) to capture the return value. After implementing,
re-verify with the same live-instrumentation technique used to find this bug (checksum
before/after broadcast across a full PPO run) — do not trust clean logs or "no errors" alone,
since that's exactly how this bug went unnoticed the first time.

## Where things are saved
- This report: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/CORRECTION_weight_sync_not_real.md`
- Debug run log (temporary instrumentation, reverted after use): `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/logs/debug_weightsync_test.log`
- Fix-candidate test scripts: written to `/tmp/`, deleted after use — not preserved (all confirmed results are captured in this report)
