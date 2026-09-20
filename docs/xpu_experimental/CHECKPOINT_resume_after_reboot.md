# CHECKPOINT — resume after machine reboot

**Written:** 2026-07-28. **Why:** the single XPU on this box wedged (even
`torch.ones(device="xpu")` hangs) during XCCL experiments; a reboot is needed. This file is the
exact state + resume steps so work continues without re-deriving anything.

---

## 0. FIRST THING after reboot — confirm the XPU is healthy

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:/home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf/bin:$PATH"
export PYTHONPATH="/home/dut7054/openrlfh_exp/OpenRLHF:$PYTHONPATH"
timeout 30 python -c "import torch; x=torch.ones(3,device='xpu'); torch.xpu.synchronize(); print('XPU OK', float(x.sum()))"
```
Must print `XPU OK 3.0` fast. If it hangs (exit 124/143), the XPU is still wedged — reboot again.

**Env notes:** venv is `work2_with_transformers_native/OpenRLHF/venv-tf` (activate/PATH only —
code lives in `openrlfh_exp/OpenRLHF`). Run XPU things with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
This box has **1 physical XPU** (`torch.xpu.device_count() == 1`).

---

## 1. Where the code is (all committed)

- Branch: `xpu/experimental-e2e-baseline`. HEAD = **`f7f93ef`** ("XPU weight-sync: add opt-in
  XCCL ... with single-XPU gloo fallback").
- **1 commit is unpushed** (`f7f93ef`) + earlier doc commits. Push needs a token (SSH/HTTPS
  blocked in-shell; user pushes manually). `git push origin xpu/experimental-e2e-baseline`
  currently diverges from remote due to prior GitHub web edits — may need
  `git push --force-with-lease` (local is the source of truth).
- Untracked data files `gsm8k_train_prompts.jsonl`, `pref_tiny.jsonl` are run artifacts — leave
  unstaged (regenerate if missing; see §5).

---

## 2. What is DONE and VERIFIED (device-free, trustworthy)

- **gloo weight-sync** (Option A): shipped, tested, works on single XPU. This is the safe default.
- **XCCL weight-sync** (Option B) is **implemented** in `openrlhf/utils/distributed_util.py`:
  - `_XcclBroadcastCommunicator.broadcast(tensor, src, stream=None)` — direct XPU broadcast, in
    place, no CPU staging.
  - `_init_xccl_process_group(master, port, rank, world_size, device)` — builds a stateless XCCL
    ProcessGroup; **device_id fix applied** (`torch.xpu.set_device(index)`, `pg.bound_device_id`,
    `_register_backend(torch.device("xpu", index), XCCL, ...)`) to avoid the "device unknown ->
    hang" warning.
  - `resolve_vllm_sync_backend`: auto = nccl(CUDA/ROCm)/gloo(else); `xccl` is opt-in via
    `--vllm.sync_backend xccl`; **falls back to gloo (warns) when `torch.xpu.device_count() < 2`**
    so single-XPU never hangs.
  - `--vllm.sync_backend` flag accepts `nccl|gloo|xccl`, default None.
- **Device-free unit tests pass** (verified after code change, 17 passed in ~4s):
  `pytest tests/test_gloo_weight_sync.py -k "resolve or communicator or stream"`.
- **Isolated 2-rank XCCL broadcast on 1 XPU worked** earlier (quick script) — but was unreliable
  under load and the FULL training loop HUNG (see §3).

## 3. What is NOT verified / the open problem

- **XCCL in a full GRPO run HANGS on a single XPU** (2 ranks share device 0; XCCL is
  device-to-device and deadlocks). The `device_id` fix is applied but **NOT yet validated** —
  it needs a **2-physical-XPU machine**. On this 1-XPU box the resolver now falls back to gloo,
  so it won't hang, but that means **XCCL itself can't be exercised here**.
- **Real XPU→XPU transfer + speed-vs-gloo: unmeasured.** Needs 2 XPUs.
- Full `pytest tests/` last completed cleanly at **82 passed** BEFORE the device wedged; after the
  wedge everything hung. **Re-run the full suite after reboot to confirm 82+ still green.**

## 4. RESUME STEPS (in order, after §0 passes)

1. **Re-confirm nothing regressed on the healthy device:**
   ```bash
   ONEAPI_DEVICE_SELECTOR=level_zero:0 python -m pytest tests/ -q -p no:cacheprovider
   ```
   Expect ~82 passed, 1 skipped (the xccl-broadcast test skips: needs >=2 XPU).

2. **Re-verify gloo E2E still works (single XPU, the shipped path):**
   ```bash
   ray stop --force; rm -rf /tmp/ray
   ONEAPI_DEVICE_SELECTOR=level_zero:0 ray start --head --num-gpus=1; sleep 3
   ONEAPI_DEVICE_SELECTOR=level_zero:0 python -m openrlhf.cli.train_ppo_ray \
     --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
     --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
     --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
     --train.colocate_all --train.colocate_actor_ref \
     --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
     --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
     --actor.model_name_or_path Qwen/Qwen2.5-0.5B \
     --reward.remote_url examples/python/math_reward_func.py \
     --data.prompt_dataset gsm8k_train_prompts.jsonl \
     --data.input_key prompt --data.label_key label --data.apply_chat_template \
     --data.max_samples 32 --data.max_len 512 \
     --train.batch_size 8 --train.micro_batch_size 4 \
     --ds.attn_implementation sdpa --ds.zero_stage 2 --ds.adam_offload \
     --ckpt.output_dir /tmp/e2e_gloo_recheck --ckpt.save_steps -1 --logger.logging_steps 1 --eval.steps -1
   ```
   (auto-selects gloo). Expect EXIT 0, Global steps, `timing/broadcast`. `ray stop --force` after.

3. **XCCL work resumes ONLY on a 2-XPU machine** — follow
   `docs/xpu_experimental/HANDOFF_xccl_xpu_to_xpu_weight_sync.md`. On this 1-XPU box, do NOT try
   to force real xccl (it hangs + can re-wedge the device). If you must sanity-check the xccl
   *code path* on 1 XPU, use a short bash `timeout` and expect it may hang — don't run it inside
   the pytest suite (guard makes the real test skip when <2 XPU, so the suite is safe).

4. **Fold in the research agent's findings** (agent id was running at reboot: "Research Intel XPU
   XCCL colocated weight-sync"). It was gathering: whether multi-rank-single-XPU XCCL is even
   supported, the exact `set_device`+`device_id` init recipe, and CCL env vars
   (`CCL_ZE_IPC_EXCHANGE`, `ZE_FLAT_DEVICE_HIERARCHY`, `CCL_ATL_TRANSPORT`,
   `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK`). If the report landed, apply relevant env/init tweaks
   to `_init_xccl_process_group` and update the HANDOFF doc.

## 5. Handy regen (if data files missing)
```bash
# gsm8k_train_prompts.jsonl (32 prompts) — see prior runs; math_reward_func expects {prompt,label}
```
(They were generated earlier this session; not committed on purpose.)

## 6. Key files
- `openrlhf/utils/distributed_util.py` — gloo + xccl communicators, resolver, group builders.
- `openrlhf/trainer/ray/ppo_actor.py` (sender) / `vllm_worker_wrap.py` (receiver) — thread backend.
- `openrlhf/cli/train_ppo_ray.py` — `--vllm.sync_backend` flag.
- `tests/test_gloo_weight_sync.py` — resolver + broadcast tests (xccl test skips <2 XPU).
- `docs/xpu_experimental/HANDOFF_xccl_xpu_to_xpu_weight_sync.md` — 2-XPU finish plan.
- `docs/xpu_experimental/reports/SUMMARY_xpu_enablement.md` — GM summary.

## 7. One-line status
gloo = done/shipped/single-XPU-safe. XCCL = implemented + device_id fix + single-XPU gloo
fallback (no hang), **pending 2-XPU validation**. Machine needed a reboot due to a wedged XPU
from XCCL experiments. Resume at §4 step 1.
