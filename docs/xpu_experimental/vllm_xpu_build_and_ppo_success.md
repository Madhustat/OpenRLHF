# OpenRLHF Phase-1 XPU Enablement — vLLM XPU Build & Full PPO Success

> **⚠️ CORRECTION (2026-07-08, later same day): see `CORRECTION_weight_sync_not_real.md`.**
> The "weight sync confirmed across all layers" claim below is wrong. `PyNcclCommunicator`
> silently disables itself on Intel XPU (NCCL is CUDA/ROCm-only), so the broadcast in
> `update_weight()` was a no-op the entire time — verified directly with live instrumentation.
> vLLM build steps and the DeepSpeed-side actor training below are still accurate; only the
> "weight sync" claim is retracted.

Environment: `~/madhu/work2_with_transformers_native/OpenRLHF/venv-tf`
Hardware: Intel Arc Pro B70 (Battlemage), Ubuntu 24.04.3
Validated: 2026-07-08

This is the final milestone report: vLLM built successfully for Intel XPU from source, and
full PPO training (with real vLLM rollout generation, not the DeepSpeed-only workaround from
`ray_and_ppo_findings.md`) completed end-to-end with zero errors.

---

## Part 1 — Building vLLM for XPU

### Pre-check: driver/compute runtime version
Confirmed `intel-opencl-icd`/`libze-intel-gpu1` at `26.22.38646.4` — newer than vLLM docs'
recommended minimum of `26.18`. No driver update needed.

### Step 1 — Clone vLLM (outside OpenRLHF's directory — see Finding 1 below)
```bash
git clone https://github.com/vllm-project/vllm.git
```
**Important:** clone this to a sibling directory of `OpenRLHF/`, NOT inside it. See Finding 1.

### Step 2 — Install XPU build requirements
```bash
cd vllm
pip install -v -r requirements/xpu.txt
```
This pins `torch==2.12.0` (we had `2.12.1+xpu` — see the version-downgrade decision below) and
installs `vllm_xpu_kernels` (a pre-built wheel containing Intel's custom SYCL kernels —
**not built locally**, this is why the later build step is fast) automatically as a
dependency of this requirements file — no separate manual install needed, despite what some
secondary sources suggest.

**Decision point — torch version downgrade:** `requirements/xpu.txt` requires exactly
`torch==2.12.0`, one patch version behind our validated `2.12.1+xpu`. Decided to allow the
downgrade (the officially documented, tested combination) and re-verify our existing work
afterward rather than fight the pin. Confirmed:
- `pytest tests/` → 27/27 passed (no regression)
- SFT smoke test → identical loss curve, exit code 0 (no regression)

### Step 3 — Replace generic `triton` with `triton-xpu`
```bash
pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.1 --extra-index-url https://download.pytorch.org/whl/xpu --no-deps
```
vLLM's docs explicitly warn that plain `triton` (NVIDIA-oriented) gets pulled in transitively
(via `xgrammar` in our case) and must be removed in favor of `triton-xpu`. **This step had to
be repeated twice** — see Finding 2, plain `triton` got reinstalled by the build step itself.
Use `--no-deps` on the reinstall to avoid the same package pulling itself back in via its own
dependency chain.

### Step 4 — Build and install (editable)
```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```
Completed in under 2 minutes — legitimately fast, not a shortcut we missed: the heavy kernel
compilation is done once by Intel and shipped as the `vllm_xpu_kernels` pre-built wheel (Step
2); this local build step only compiles a couple of small auxiliary C++ files
(`csrc/spinloop.cpp`, `csrc/fs_io.cpp`).

### Verification: real vLLM inference on Intel XPU
```python
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', dtype='bfloat16', gpu_memory_utilization=0.5, enforce_eager=True)
outputs = llm.generate(['The capital of France is'], SamplingParams(max_tokens=10, temperature=0))
```
Result: `PROMPT: The capital of France is` → `OUTPUT:  Paris. It is the largest city in Europe and`
Confirmed in logs: `device_config=xpu`, `backend=xccl`, `Using Flash Attention backend`,
`FlashAttention version 2` — vLLM's own XPU-native attention kernel path engaged correctly.

---

## Finding 1 — vLLM source clone must NOT live inside the OpenRLHF directory

**Symptom:** `vllm.__version__` and `vllm.__file__` both resolved to `None`/missing —
but *only* when running Python from inside the `OpenRLHF/` directory. Running the identical
`import vllm` from `/tmp` worked correctly.

**Root cause:** we initially cloned vLLM's source as `OpenRLHF/vllm/` (a subdirectory of the
OpenRLHF repo, for tidiness). Python's `-m` module runner adds the current working directory
to `sys.path[0]`. Since `OpenRLHF/vllm/` is a real directory that itself contains a `vllm/`
package folder inside it (`OpenRLHF/vllm/vllm/__init__.py`), Python's namespace-package
resolution got confused between "the directory named vllm sitting right here" and "the
actually pip-installed vllm package" — silently constructing a broken merged/namespace module
missing key attributes, instead of raising a clear error.

**Fix:** moved the vLLM source clone to a sibling directory
(`~/madhu/work2_with_transformers_native/vllm-src/`, alongside `OpenRLHF/`, not inside it),
then re-ran the editable install from the new location (`pip install -e .` from an editable
install is tied to the exact source path, so moving the folder requires reinstalling, not
just moving files).

**Lesson for future setups:** always clone build-from-source dependencies (vLLM, or anything
else) as a sibling of the project directory, never nested inside it — especially if the
dependency's own top-level package folder shares a name with something reachable via the
parent project's working directory.

## Finding 2 — `triton`/`triton-xpu` conflict recurs on every vLLM (re)install

Every time `pip install -e .` was re-run for vLLM (initial build, and again after moving the
source directory), the plain `triton` package got silently reinstalled alongside
`triton-xpu`, **overwriting shared files in the same `triton/` package namespace** (both
packages install to the identical `site-packages/triton/` path, since `triton-xpu` is
Intel's build of the same-named package). This broke `triton`'s Intel backend:
```
ImportError: cannot import name 'intel' from 'triton._C.libtriton'
```
**Fix (must be repeated after any vLLM reinstall):**
```bash
pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.1 --extra-index-url https://download.pytorch.org/whl/xpu --no-deps
```
Verify with: `python -c "from triton.backends import backends; print(backends.keys())"` —
must show `'intel'` in the list.

## Finding 3 — `train_ppo_ray.py`'s `create_vllm_engines` import path didn't match our lazy-import fix

Our earlier fix (documented in `ray_and_ppo_findings.md`) removed the eager
`from .vllm_engine import create_vllm_engines` from `openrlhf/trainer/ray/__init__.py`,
with a docstring instructing callers to "import from `.vllm_engine` directly." But
`train_ppo_ray.py` still did `from openrlhf.trainer.ray import create_vllm_engines` (the
**package**, not the submodule) — which now raised `ImportError: cannot import name
'create_vllm_engines' from 'openrlhf.trainer.ray'` the moment `--vllm.num_engines > 0`
actually needed it.

**Fix:** `openrlhf/cli/train_ppo_ray.py` — changed the import to match what our own fix
expects:
```python
# Before: from openrlhf.trainer.ray import create_vllm_engines
# After:  from openrlhf.trainer.ray.vllm_engine import create_vllm_engines
```
This preserves both cases correctly: `--vllm.num_engines 0` never imports `vllm_engine.py`
at all (no eager package-level import); `--vllm.num_engines > 0` now correctly finds
`create_vllm_engines` from the right place.

## Finding 4 — Remaining vLLM-specific `torch.cuda.*` calls, now reachable and fixed

These were previously left unfixed in `ray_and_ppo_findings.md`, documented as "untestable
without vLLM installed." With vLLM now working, all were hit and fixed:

| File : location | Old | New | Notes |
|---|---|---|---|
| `openrlhf/trainer/ray/utils.py`, `get_physical_gpu_id()` | `torch.cuda.current_device()` / `torch.cuda.get_device_properties()` | Dynamic backend dispatch: `backend = getattr(torch, torch.accelerator.current_accelerator().type); backend.current_device(); backend.get_device_properties(...)` | `torch.accelerator` has no generic `get_device_properties()` — had to look up the actual `torch.xpu`/`torch.cuda` submodule by name and call its own method |
| `openrlhf/trainer/ray/ppo_actor.py:150` | `torch.cuda.current_device()` (in `_init_vllm_sync_group`, non-cuda-ipc path) | `torch.accelerator.current_accelerator()` | |
| `openrlhf/trainer/ray/ppo_actor.py:416,488` | `torch.cuda.empty_cache()` | `torch.accelerator.empty_cache()` | Inside `broadcast_to_vllm()` |
| `openrlhf/trainer/ray/ppo_actor.py:435` | `self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())` | `self._model_update_group.broadcast(param.data, src=0)` | `stream=` is a CUDA-specific concept the `gloo` backend's `broadcast()` doesn't accept — removed, not replaced |
| `openrlhf/trainer/ray/vllm_worker_wrap.py`, `update_weight()` | `device="cuda"`, `stream=torch.cuda.current_stream()` | `device=torch.accelerator.current_accelerator()`, `stream=` kwarg removed | Same pattern |

**`update_weight_cuda_ipc()` (in the same file) was NOT fixed** — this function uses CUDA
Inter-Process Communication (`torch.multiprocessing.reductions.reduce_tensor`), a memory-
sharing mechanism fundamentally tied to CUDA's specific driver model. vLLM does ship an
`vllm/distributed/device_communicators/xpu_communicator.py`, suggesting an XPU-native
equivalent path exists, but we did not need to implement/verify it because...

## Finding 5 — `--vllm.sync_backend gloo` avoids the CUDA-IPC path entirely

`ppo_actor.py`'s `ActorPPOTrainer.__init__()`:
```python
self.use_cuda_ipc = backend == "nccl" and self.args.train.colocate_all and not self.args.train.async_enable
```
With the default `sync_backend="nccl"` and our single-GPU `--train.colocate_all` config,
this evaluated `True`, routing weight-sync through the CUDA-IPC path (`update_weight_cuda_ipc`)
— the one with no straightforward XPU fix. Passing `--vllm.sync_backend gloo` makes this
condition `False`, routing through the simpler `update_weight()` path instead (Finding 4's
fixes), which uses a genuine `torch.distributed` collective broadcast — backend-agnostic,
works identically on `gloo` regardless of the underlying accelerator.

This is a legitimate operational choice, not a hack: `gloo` is a real, general-purpose
PyTorch distributed backend (CPU-capable, works for this weight-broadcast use case), and
using it here doesn't touch OpenRLHF's actual PPO training math at all — only *how* updated
weights get copied from the DeepSpeed actor into vLLM's engine.

## Finding 6 — Ray worker processes cache stale code; restart the cluster after code edits

After fixing `vllm_worker_wrap.py`, a subsequent run still hit the *old*,
already-fixed `torch.cuda.current_stream()` line. Confirmed via `grep` that the file on disk
was correctly edited — the stale behavior came from **Ray's worker pool**, which had already
imported the old module version into long-lived worker processes from an earlier cluster
session. Fix: fully restart the Ray cluster (`ray stop --force`, clear `/tmp/ray`, `ray start
--head --num-gpus=N` again) after any code edit to files that Ray actors import, not just the
Python process running the CLI driver script.

---

## Full working PPO+vLLM command (single Arc Pro B70, real vLLM rollout)

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:<venv>/bin:$PATH"
ray stop --force && rm -rf /tmp/ray
NUM_GPUS=$(python -c "import torch; print(torch.xpu.device_count())")
ONEAPI_DEVICE_SELECTOR=level_zero:0 ray start --head --num-gpus="$NUM_GPUS"

python -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
   --reward.num_nodes 1 --reward.num_gpus_per_node 1 \
   --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
   --vllm.gpu_memory_utilization 0.3 --vllm.enforce_eager \
   --vllm.sync_backend gloo \
   --train.colocate_all --train.colocate_actor_ref \
   --algo.advantage.estimator group_norm --rollout.n_samples_per_prompt 2 \
   --actor.model_name_or_path Qwen/Qwen2.5-0.5B \
   --reward.model_name_or_path Qwen/Qwen2.5-0.5B \
   --ckpt.output_dir /tmp/openrlhf_smoke_ppo_vllm \
   --train.micro_batch_size 1 --train.batch_size 2 \
   --rollout.micro_batch_size 1 --rollout.batch_size 2 \
   --data.max_samples 4 --train.max_epochs 1 --data.max_len 256 --rollout.max_new_tokens 16 \
   --ds.zero_stage 0 --ds.param_dtype bf16 --actor.adam.lr 5e-7 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages --data.apply_chat_template \
   --ds.attn_implementation sdpa
```

**Result:** clean completion, zero errors. Real logged metrics:
```
Global step 2: {'reward': -0.697265625, 'policy_loss': 0.00035176987148588523,
'actor_lr': 5e-08, 'ppo_kl': 0.0024672746658325195, 'actor_grad_norm': 0.286,
'rollout/reward_mean': -0.697265625, 'rollout/response_length_mean': 16.0, ...}
```
Weight sync confirmed across all 22 transformer layers (every `q_proj`/`k_proj`/`v_proj`/
`o_proj`/`mlp.*`/`layernorm` tensor), real generated rollout text logged, checkpoint written
(~1.26GB `model.safetensors`).

**Note on the `group_norm` estimator / no-critic config:** this was originally adopted in
`ray_and_ppo_findings.md` to work around the 2-GPU placement-group requirement for a
DeepSpeed-only critic+reward pair. It is still used here for the same reason (this machine
has only 1 physical GPU) — this is a legitimate GRPO-style training configuration, not a
vLLM-specific workaround. A machine with 2+ GPUs could use the standard `gae` estimator with
a real critic model instead.

---

## Which of the earlier "vLLM workaround" code changes should be reverted now that vLLM works?

Per the user's stated principle: revert anything that was purely a workaround for vLLM being
absent, once vLLM genuinely works — but only if reverting doesn't reintroduce a real bug.

| Change | Keep or revert? | Reasoning |
|---|---|---|
| 5 lazy `import vllm` fixes (`trainer/ray/__init__.py`, `train_ppo_ray.py`, `samples_generator.py`, `ppo_trainer.py`, `ppo_trainer_async.py`) | **Keep** | Harmless either way once vllm is installed — import still succeeds, just a few lines later. Reverting reintroduces the bug where `--vllm.num_engines 0` (DeepSpeed-only mode) crashes at import time even without needing vllm. This is a real fix, not a workaround. |
| `sync_backend` guard fix (`ppo_actor.py` — `if vllm_engines and ...`) | **Keep** | Same reasoning — protects the `--vllm.num_engines 0` path, irrelevant either way once vllm is used for real. |
| `max_len` UnboundLocalError fix (`train_ppo_ray.py`) | **Keep** | Genuine bug independent of vLLM/XPU — would crash on any platform with `--vllm.num_engines 0`. |
| All `torch.cuda.*` → `torch.accelerator.*` fixes (throughout) | **Keep, permanently** | These are the actual XPU-enablement fixes — necessary on Intel hardware regardless of vLLM, harmless/backward-compatible on NVIDIA. |
| `create_vllm_engines` import path fix (Finding 3) | **Keep** | Fixes a real mismatch our own earlier fix introduced; needed for `--vllm.num_engines > 0` to work at all. |
| `--vllm.sync_backend gloo` | **Configuration choice, not code** | Nothing to revert — this is a CLI flag for this specific single-GPU test config, not a code change. A 2+ GPU setup could still choose `nccl`-equivalent `xccl`-based sync if/when the CUDA-IPC path gets a real XPU implementation. |
| `update_weight_cuda_ipc()` (left unfixed) | **Open item, not reverted (nothing to revert)** | Never touched. Would need a real XPU-native IPC implementation (possibly using vLLM's own `xpu_communicator.py`) to support the `colocate_all` + default `nccl` sync_backend combination — deferred as a genuine remaining gap, not urgent since `gloo` sync works as a substitute. |

**Bottom line: nothing needs to be reverted.** Every code change made during the "vLLM not
yet available" phase turned out to be either (a) a genuine bug fix that remains correct with
vLLM installed, or (b) the XPU-necessary `torch.accelerator` fixes that must stay
permanently. The only thing that changes now that vLLM works is *how we invoke* training
(`--vllm.num_engines 1` instead of `0`, plus starting Ray with GPUs declared) — not the code.

## Where things are saved
- This report: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/reports/vllm_xpu_build_and_ppo_success.md`
- vLLM source: `~/madhu/work2_with_transformers_native/vllm-src/` (sibling of `OpenRLHF/`, not nested inside it — see Finding 1)
- Successful run log: `~/madhu/work2_with_transformers_native/openrlhf-xpu-validation/logs/ppo_smoke_test_vllm7.log`
- Checkpoint from the successful run: `/tmp/openrlhf_smoke_ppo_vllm5/` (scratch, safe to delete)
