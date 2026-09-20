# Code Review Reference — Every File Changed for Intel XPU Enablement

Environment: `~/madhu/work2_with_transformers_native/OpenRLHF/`
Generated: 2026-07-08, from `git diff` against upstream OpenRHLF commit `3f8ae08`

This is a per-file, per-line reference for reviewing every change made to port OpenRLHF to
Intel XPU. **16 files modified, 3 test files added.** All line numbers below are from the
CURRENT (modified) file, not upstream — use them directly with your editor's "go to line."

For narrative context (why each change was needed, what broke without it), see:
- `MASTER_setup_and_fixes_guide.md` — original SFT-era fixes
- `ray_and_ppo_findings.md` — Ray/PPO-era fixes, found before vLLM was built
- `vllm_xpu_build_and_ppo_success.md` — vLLM-era fixes, found once vLLM was installed
- `rlhf_methods_coverage.md` — the one RM bug (fix #13)
- `CORRECTION_weight_sync_not_real.md` — **important**: the weight-sync-related lines in
  `ppo_actor.py` and `vllm_worker_wrap.py` below are necessary but NOT sufficient — they fix
  the *device* used, but the actual weight broadcast is still silently broken for a different
  reason (NCCL unavailable on XPU). Read that report before reviewing those two files.

Reproduce this diff yourself at any time with: `cd OpenRLHF && git diff`

---

## Summary table — all 16 modified files

| # | File | Lines changed | Category |
|---|---|---|---|
| 1 | `openrlhf/cli/train_ppo_ray.py` | 8 (removed), 53-55 | Import fix + genuine bug fix |
| 2 | `openrlhf/cli/train_rm.py` | 81-86 | Genuine bug fix (not XPU-specific) |
| 3 | `openrlhf/models/ring_attn_utils.py` | 3-8, 13-41, 76 | Dependency replacement + device fix |
| 4 | `openrlhf/trainer/ppo_trainer.py` | 17 (removed), 311-312 | Lazy import |
| 5 | `openrlhf/trainer/ppo_trainer_async.py` | 10 (removed), 255-256 | Lazy import |
| 6 | `openrlhf/trainer/ppo_utils/replay_buffer.py` | 40, 104, 129, 159, 165 | Device fix (×5) |
| 7 | `openrlhf/trainer/ppo_utils/samples_generator.py` | 6-8 (removed), 11-16, 71-72, 89-90, 115-116, 130-131, 215-216 | Lazy import (×5 sites) |
| 8 | `openrlhf/trainer/ray/__init__.py` | 2-11 | Lazy import (package-level) |
| 9 | `openrlhf/trainer/ray/launcher.py` | 64-65, 136, 190 | Device fix (×3) |
| 10 | `openrlhf/trainer/ray/ppo_actor.py` | 150, 173, 416, 435, 488, 503-505, 601, 605-606, 628 | Device fix (×7) + guard bug fix + stream kwarg removal |
| 11 | `openrlhf/trainer/ray/ppo_critic.py` | 83, 269, 288, 292-293 | Device fix (×4) |
| 12 | `openrlhf/trainer/ray/utils.py` | 46-48 | Device fix (dynamic dispatch) |
| 13 | `openrlhf/trainer/ray/vllm_worker_wrap.py` | 40, 46 | Device fix + stream kwarg removal |
| 14 | `openrlhf/utils/deepspeed/deepspeed.py` | 97, 106-108, 400, 614, 633-634 | Device fix (×5) |
| 15 | `openrlhf/utils/distributed_util.py` | 10 | Device fix |
| 16 | `openrlhf/utils/loss_utils.py` | 67 | Device fix (fallback logic) |

**+ 3 new test files** (not modifications): `tests/test_advantage_estimators.py`,
`tests/test_loss_aggregation_xpu.py`, `tests/test_xccl_backend.py`

**Recurring pattern across ~40 of the individual line changes:** `torch.cuda.*` →
`torch.accelerator.*` (PyTorch's generic, device-agnostic accelerator API — resolves to XPU on
this hardware, CUDA on NVIDIA, with the exact same code). This is the single dominant category;
files 6, 9, 10 (partially), 11, 14, 15 are entirely or almost entirely this one pattern.

---

## 1. `openrlhf/cli/train_ppo_ray.py`

**Line 8 (removed):** `from openrlhf.trainer.ray import create_vllm_engines` — deleted.
**Lines 53-55 (current):**
```python
    vllm_engines = None
    max_len = args.data.max_len
    if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
        from openrlhf.trainer.ray.vllm_engine import create_vllm_engines

        if args.train.colocate_all and not args.train.async_enable:
```
**Two independent changes bundled here:**
- Import moved from package-level (file 8, `trainer/ray/__init__.py`) directly to
  `trainer/ray/vllm_engine` submodule, and made lazy (only runs inside the
  `num_engines > 0` branch). Necessary companion to fix #8 below.
- **Genuine bug, not XPU-specific:** `max_len = args.data.max_len` moved OUTSIDE the
  `if args.vllm.num_engines...` block. Previously only defined inside that block but used
  unconditionally later in the function (`PPOTrainer.remote(..., max_len=max_len, ...)`) —
  would raise `UnboundLocalError` on any platform when running with `--vllm.num_engines 0`.

## 2. `openrlhf/cli/train_rm.py`

**Lines 81-86 (current):**
```python
    else:
        # Used for calculating mean/std for reward normalization. int(n * 0.01) rounds
        # down to 0 for small n (e.g. any train_data under 100 rows, common in smoke
        # tests), which then makes dataset.map(..., remove_columns=...) drop all
        # columns on the empty result and crash downstream with "Column 'prompt'
        # doesn't exist" instead of a clear error. Floor at 1 row.
        eval_data = train_data.select(range(min(args.data.max_samples, max(1, int(len(train_data) * 0.01)))))
```
**Genuine bug, not XPU-specific.** Was: `int(len(train_data) * 0.01)` with no floor — rounds
to 0 whenever `train_data` has under 100 rows, producing an empty eval dataset that later
crashes with a confusing, unrelated-looking error. Fixed by wrapping in `max(1, ...)`.

## 3. `openrlhf/models/ring_attn_utils.py`

**Lines 3-8 (current, replacing 2 removed lines):**
```python
from einops import rearrange
from transformers.modeling_flash_attention_utils import (
    _index_first_axis as index_first_axis,
    _pad_input as pad_input,
    _unpad_input as unpad_input,
)
```
Removed: `from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input`
and `from flash_attn.utils.distributed import all_gather`. **Dependency swap**, not a bug fix —
`flash_attn` isn't installable/usable in this setup; these 3 helper functions have equivalent,
signature-compatible implementations already inside `transformers` (verified at
`transformers==5.13.0`).

**Lines 13-41 (current, new code):** a vendored `_AllGatherFunc(torch.autograd.Function)`
class (~20 lines) plus `all_gather = _AllGatherFunc.apply`. `transformers` doesn't ship a
replacement for `flash_attn`'s `all_gather`, so this one had to be hand-written. Pure
`torch.distributed` (`all_gather_into_tensor`/`reduce_scatter_tensor`), no CUDA-specific code —
review this one most carefully since it's the only genuinely new logic in this file, not just an
import swap.

**Line 76 (current):**
```python
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.accelerator.current_accelerator())
```
Was `device=torch.cuda.current_device()`. Standard device-fix pattern.

## 4. `openrlhf/trainer/ppo_trainer.py`

**Line 17 (removed):** `from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call`.
**Lines 311-312 (current):**
```python
        if self.args.vllm.enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            # Wake up only weights for weight sync (not KV cache)
```
Import moved from module-level to lazy, inside the one `if` branch that uses it. Purpose:
`vllm` package is optional (PPO can run with `--vllm.num_engines 0`); an eager import here
would fail/block even for runs that never touch vLLM.

## 5. `openrlhf/trainer/ppo_trainer_async.py`

**Line 10 (removed):** same import as file 4, removed.
**Lines 255-256 (current):** same lazy-import pattern, inside `broadcast_to_vllm()`.

## 6. `openrlhf/trainer/ppo_utils/replay_buffer.py`

Five independent `torch.cuda.*` → `torch.accelerator.*` device fixes, all in `NaiveReplayBuffer`:
- **Line 40:** `self.target_device = torch.accelerator.current_accelerator()` (was
  `torch.device(f"cuda:{torch.cuda.current_device()}")`)
- **Line 104:** `num_steps = torch.tensor(..., device=torch.accelerator.current_accelerator())`
- **Line 129:** `num_microbatches = torch.tensor(..., device=torch.accelerator.current_accelerator())`
- **Line 159:** `global_valid_sample_num = torch.tensor(..., device=torch.accelerator.current_accelerator())`
- **Line 165:** `global_num_tokens = torch.tensor(..., device=torch.accelerator.current_accelerator())`

All 5 are the same mechanical pattern — low review risk, just confirm each call site still
passes a valid device argument to `torch.tensor(...)`.

## 7. `openrlhf/trainer/ppo_utils/samples_generator.py`

**Lines 6-8 (removed):** `from vllm import SamplingParams` and
`from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call` — both deleted from
module level.
**Lines 11-16 (current):** replaced with an explanatory comment (no code) about why the
imports moved.
**5 lazy-import insertion sites** (same 2-line pattern each — blank line + import inside an
existing `if`/`try` block, no other logic changed):
- **Lines 71-72:** inside `if self.args.vllm.enable_sleep:` (in one method)
- **Lines 89-90:** inside `if self.args.vllm.enable_sleep:` (in a `finally:` block)
- **Lines 115-116:** inside `if self.args.vllm.enable_sleep:` (another call site)
- **Lines 130-131:** inside `if self.args.vllm.enable_sleep:` (another call site)
- **Lines 215-216:** `from vllm import SamplingParams` inside `_dispatch_prompts_to_vllm()`,
  right before `sampling_params = SamplingParams(...)` is constructed

Same rationale throughout: `vllm` is optional, must not be imported at module load time.

## 8. `openrlhf/trainer/ray/__init__.py`

**Lines 2-11 (current, replacing 6 removed lines):**
```python
#
# create_vllm_engines/batch_vllm_engine_call require the `vllm` package, which
# is optional - PPO/GRPO training can run with `--vllm.num_engines 0`
# (DeepSpeed-only rollout, no vLLM), and vllm may not be installed at all in
# that case. Callers must import from .vllm_engine directly, inside the code
# path that actually uses vLLM (see train_ppo_ray.py's `train()`), rather than
# importing this package eagerly - an unconditional import here would block
# train_ppo_ray.py entirely even for runs that never touch vLLM. Same class of
# bug as the flash_attn import fix in ring_attn_utils.py (see
# MASTER_setup_and_fixes_guide.md Part 3.1).
```
Removed: `from .vllm_engine import batch_vllm_engine_call, create_vllm_engines` and the
`__all__` list re-exporting them. **This is the root fix that files 1, 4, 5, 7 above are all
downstream companions of** — review this one first, then trace forward to its callers.

## 9. `openrlhf/trainer/ray/launcher.py`

Three device fixes:
- **Lines 64-65:**
  ```python
      def empty_cache(self) -> None:
          torch.accelerator.empty_cache()
          torch.accelerator.synchronize()
  ```
  (was `torch.cuda.empty_cache()` / `torch.cuda.synchronize()`)
- **Line 136:** `device = torch.accelerator.current_accelerator()` in `ReferenceModelActor`
- **Line 190:** `device = torch.accelerator.current_accelerator()` in `RewardModelActor`

## 10. `openrlhf/trainer/ray/ppo_actor.py`

**Largest file by change count — 9 hunks.** Mix of mechanical device fixes and 2 more
substantial changes; review those two more carefully.

- **Line 150:** `stateless_init_process_group(master_address, master_port, 0, world_size, torch.accelerator.current_accelerator())`
  (was `torch.cuda.current_device()`) — in `_init_vllm_sync_group()`
- **Line 173:** `device = torch.accelerator.current_accelerator()` — in `ppo_train()`
- **Line 416:** `torch.accelerator.empty_cache()` — in `broadcast_to_vllm()`
- **Line 435 — review carefully:**
  ```python
                      self._model_update_group.broadcast(param.data, src=0)
  ```
  Was: `self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())`.
  The `stream=` kwarg was **removed, not replaced** — `gloo` backend's `.broadcast()` doesn't
  accept a `stream` argument. **Important:** per `CORRECTION_weight_sync_not_real.md`, this
  broadcast call is currently a silent no-op on XPU for an unrelated reason (NCCL unavailable) —
  this line's fix is necessary but the broadcast still doesn't do anything real yet.
- **Line 488:** `torch.accelerator.empty_cache()` — in a cache-reset path
- **Lines 503-505 — review carefully, genuine logic bug fix:**
  ```python
          # Only relevant when vLLM engines are actually in use - vllm may not be
          # installed at all when running with --vllm.num_engines 0.
          if vllm_engines and getattr(args.vllm, "sync_backend", "nccl") == "nccl":
  ```
  Was: `if getattr(args.vllm, "sync_backend", "nccl") == "nccl":` (no `vllm_engines and` guard).
  Without the added guard, this branch executed `import vllm` unconditionally whenever
  `sync_backend` defaulted to `"nccl"` — even when `vllm_engines` is `None`/empty (i.e.
  `--vllm.num_engines 0`, vLLM disabled entirely). Real bug independent of XPU.
- **Line 601, 605-606:** `torch.accelerator.empty_cache()` / `torch.accelerator.synchronize()`
  in `fit()`, straightforward device fixes
- **Line 628:** `device = torch.accelerator.current_accelerator()` in the value-generation path

## 11. `openrlhf/trainer/ray/ppo_critic.py`

Four device fixes, all mechanical, mirroring the critic-side equivalents of file 10's actor-side
changes:
- **Line 83:** `device = torch.accelerator.current_accelerator()` in `ppo_train()`
- **Line 269:** `device = torch.accelerator.current_accelerator()` in value generation
- **Line 288, 292-293:** `torch.accelerator.empty_cache()` / `.synchronize()` in `fit()`

## 12. `openrlhf/trainer/ray/utils.py`

**Lines 46-48 (current) — review carefully, the one non-mechanical device fix in this file:**
```python
    backend = getattr(torch, torch.accelerator.current_accelerator().type)
    device = backend.current_device()
    props = backend.get_device_properties(device)
```
Was:
```python
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
```
`torch.accelerator` (the generic API) has no `get_device_properties()` method — only the
device-specific submodules (`torch.xpu`, `torch.cuda`) do. This dynamically looks up the right
submodule by name (`torch.xpu` on this hardware) and calls its own method. Slightly more
fragile than the other fixes in this codebase (relies on `torch.<type>` naming matching
`torch.accelerator.current_accelerator().type` exactly) — worth double-checking if porting to
a new accelerator type in the future.

## 13. `openrlhf/trainer/ray/vllm_worker_wrap.py`

**Line 40:**
```python
        weight = torch.empty(shape, dtype=dtype, device=torch.accelerator.current_accelerator())
```
Was `device="cuda"` (hardcoded string).

**Line 46:**
```python
            self._model_update_group.broadcast(weight, src=0)
```
Was `self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())` —
same `stream=` removal as file 10's line 435, same important caveat: this broadcast call is
currently a silent no-op on XPU (see `CORRECTION_weight_sync_not_real.md`), for a reason
unrelated to this particular fix.

## 14. `openrlhf/utils/deepspeed/deepspeed.py`

Five device fixes in `DeepspeedStrategy`:
- **Line 97:** `torch.accelerator.set_device_index(local_rank)` (was `torch.cuda.set_device(local_rank)`)
- **Lines 106-108 — review carefully:**
  ```python
          self.ds_device_mesh = init_device_mesh(
              torch.accelerator.current_accelerator().type,
              (dp_size, self.ring_attn_size, self.ds_tensor_parallel_size),
              mesh_dim_names=("dp", "sp", "tp"),
          )
  ```
  Was: `init_device_mesh("cuda", (...), mesh_dim_names=(...))` — hardcoded `"cuda"` string
  literal replaced with the dynamically-resolved accelerator type string (`"xpu"` on this
  hardware). Also reformatted from one line to multiple for readability — pure formatting,
  no logic change beyond the string source.
- **Line 400:** `torch.accelerator.empty_cache()` in a tensor-parallel cleanup path
- **Line 614:** `data = data.to(torch.accelerator.current_accelerator())` in `all_reduce()`
- **Lines 633-634:** `torch.zeros_like(data).to(torch.accelerator.current_accelerator())` /
  `data.to(torch.accelerator.current_accelerator())` in `all_gather()`

## 15. `openrlhf/utils/distributed_util.py`

**Line 10:**
```python
    torch.accelerator.synchronize()
```
Was `torch.cuda.synchronize()`, inside `torch_dist_barrier_and_cuda_sync()`. Single-line,
lowest-risk change in the whole set.

## 16. `openrlhf/utils/loss_utils.py`

**Line 67 — review carefully, this is a fallback/availability-check fix, not a pure rename:**
```python
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
```
Was: `device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"`.
`torch.cuda.is_available()` returns `False` on this hardware (no NVIDIA GPU present) — so this
fallback was previously ALWAYS silently choosing `"cpu"`, not XPU, even though a working
accelerator was available. That silent wrong choice later caused a distinct downstream crash
(`RuntimeError: No backend type associated with device type cpu` in a `dist.all_reduce()` call
over the `xccl` group). Worth confirming this pattern — `is_available()` gated fallback logic —
doesn't appear elsewhere unfixed.

---

## New test files (not modifications, no upstream equivalent to diff against)

- `tests/test_advantage_estimators.py` — 9 pure-math CPU tests for PPO advantage-estimator
  formulas (`rloo`, `group_norm`, `dr_grpo`, `reinforce_baseline`, `reinforce`, `gae`,
  `get_cumulative_returns`). No device dependency.
- `tests/test_loss_aggregation_xpu.py` — 8 tests, same formulas as upstream's
  `test_loss_aggregation.py` but with real `device='xpu'` tensors.
- `tests/test_xccl_backend.py` — 2 tests exercising a real `xccl` process-group
  init + `all_reduce()` on XPU hardware; documents the `libze-dev` system dependency in its
  module docstring.

## Suggested code-review order

1. **File 8** (`trainer/ray/__init__.py`) first — it's the root cause the lazy-import changes
   in files 1, 4, 5, 7 all exist to support. Understanding it first makes those four trivial.
2. **File 3** (`ring_attn_utils.py`) — the one file with genuinely new logic (`_AllGatherFunc`),
   not just a mechanical device/import swap.
3. **Files 1 and 10** — contain the 2 non-XPU-specific logic bugs (`max_len` UnboundLocalError,
   `vllm_engines and` guard) mixed in with device fixes; isolate those from the mechanical
   changes around them.
4. **File 2** (`train_rm.py`) — standalone, genuine bug, easiest to review in isolation.
5. **File 12** (`utils.py`) — the one fix using dynamic `getattr` dispatch instead of a direct
   rename; worth confirming the naming-convention assumption it relies on.
6. **File 16** (`loss_utils.py`) — the one fix where the OLD code had a silent-wrong-fallback
   bug, not just a CUDA-specific call.
7. **Everything else** (files 4-7, 9, 11, 13-15, and the remaining hunks in file 10) — the
   repetitive `torch.cuda.*` → `torch.accelerator.*` pattern, lowest individual risk, but worth
   a quick pass to confirm no call site was missed or mismapped.
8. **Files 10 and 13's `stream=` removals specifically** — cross-reference with
   `CORRECTION_weight_sync_not_real.md` before concluding these are "done"; they're correct
   fixes but don't make PPO weight sync actually work yet.
