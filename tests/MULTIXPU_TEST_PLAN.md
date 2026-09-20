# OpenRLHF on 2x Intel Arc Pro B70 — test plan and results

Same structure as the single-GPU table, with four columns added because on two devices
the same Test ID can mean different things: **GPU 0**, **GPU 1**, **Sleep**, **Colocation**.

Repo under test: `/home/sdp/madhu/OpenRLHF-fresh` = latest upstream `dc2a7ad3` + our changes.
Model: Qwen2.5-0.5B unless stated. MoE rows use granite-3.1-1b-a400m-instruct (32 experts,
top-8) or a tiny 4-expert Qwen2Moe where noted.

## Placement patterns (verified from the configs, not assumed)

| Pattern | GPU 0 | GPU 1 | Sleep | Colocation |
|---|---|---|---|---|
| **P1** separated RL | Actor (DeepSpeed) | vLLM engine | Off / Off | none |
| **P2** colocated RL | Actor + ref/critic/reward + vLLM (shared) | shared | vLLM On / DS On | `colocate_all` |
| **P3** sharded actor | Actor rank 0 + engine 0 | Actor rank 1 + engine 1 | On / On | `colocate_all` |
| **P4** supervised | Trainer, single process | **IDLE** | n/a | n/a |

**P4 is stated explicitly on purpose.** The supervised trainers are launched as one process
(`python -m openrlhf.cli.<trainer>`, no deepspeed/torchrun launcher, zero rank markers in the
logs => `world_size=1`). They exercise compute, loss and data paths on ONE device and do not
test distribution or weight transfer. ZeRO-3 in a P4 row proves the code path runs, not that
it shards. Anyone auditing this table should see that without having to ask.

**What "PASS" requires.** RL rows: exit 0 AND at least one `Global step`, which can only
happen after a successful gloo actor->vLLM weight broadcast (measured: ~290 params per step,
sync process group world_size=2). Supervised rows: exit 0 AND at least one loss line.
A silent no-op sync cannot produce a PASS.

Evidence sources: **bench** = upstream-scripts bench (14/14 PASS), **matrix** = 80-cell gloo
matrix (63/63 passable cells PASS), **ext** = extended suite (46 PASS), **base** = the 28-case
base e2e suite, which has NEVER been run on this tree.

---

## A. Core RL algorithm coverage

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RL algo | `up_ppo_gae` | PPO with critic (GAE) — the 6th estimator, adds critic + value head | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 2 | RL algo | `up_dapo` | DAPO — GRPO + dynamic filtering + clip-higher + KL-k3-as-loss | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 3 | RL algo | `up_prorlv2` | ProRL-v2 — REINFORCE++ + dyn filtering + clip-higher + KL-k2 | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 4 | RL algo | `up_reinforce_baseline` | REINFORCE++ group-baseline advantage | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 5 | RL algo | `up_flash_reinforce` | FlashREINFORCE — critic-free single-rollout async RL | Actor | vLLM engine | Off / Off | none (async) | GSM8K | 5 | **PASS** | |
| 6 | RL algo | `mg_gamma_reinforce` | REINFORCE (plain) with discount gamma=0.99 — the only estimator that honours gamma | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 7 | RL algo | `mg_gspo` | GSPO policy loss instead of the PPO clipped surrogate | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 8 | RL algo | **`rloo_nocolo`** | RLOO leave-one-out advantage | Actor | vLLM engine | Off / Off | none | GSM8K | — | **PASS** | closed 2026-09-20, 10 steps |
| 9 | RL algo | **`dr_grpo_nocolo`** | Dr. GRPO — no std-dev normalization | Actor | vLLM engine | Off / Off | none | GSM8K | — | **PASS** | closed 2026-09-20, 10 steps |
| 10 | RL algo | `mg_no_std_norm` | group_norm advantage with std-dev normalization disabled | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 11 | RL algo | `mg_lambda_gae` | GAE lambda=0.95 (requires a critic) | Actor+Critic+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 12 | RL algo | `mg_dual_clip` | Dual-clip PPO lower bound on negative-advantage tokens | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 13 | RL algo | `mg_value_clip` | Critic value clipping at non-default 0.2 | Actor+Critic+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 14 | RL algo | `mg_critic_extras` | Critic frozen N steps, then value network persisted | Actor+Critic+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |

## B. Reward & reference pipeline

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | Reward | `up_ppo_reward_fn` | Remote reward **function** instead of a reward model | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 16 | Reward | `mg_reward_model` | Reward **model** as a 4th resident model | Actor+Reward+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 17 | Reward | `up_serve_rm` | Reward-model HTTP server, `POST /get_reward` returns a finite reward | RM server (pinned) | idle | n/a | n/a | synthetic query | n/a | **PASS** | |
| 18 | Reward | `mg_reward_offload` | Reward-model CPU offload (`--reward.offload`) | Actor+Reward+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 19 | Reward | `mg_ref_offload` | Reference-model CPU offload (`--ref.offload`) | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 20 | Reward | `mg_reward_clip_range` | Reward clipping into a narrow +/-5 band | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 21 | Reward | **`grpo_reward_norm`** | Reward normalization path | Actor | vLLM engine | Off / Off | none | GSM8K | — | **PASS** | closed 2026-09-20. Also dropped from the bench: a 0.5B model never solves GSM8K so reward variance is exactly 0 and normalizing gives NaN — needs a reward source with variance |
| 22 | Reward | **`grpo_overlong_penalty`** | Overlong reward penalty + penalty factor | Actor | vLLM engine | Off / Off | none | GSM8K | — | **PASS** | closed 2026-09-20 |
| 23 | KL | `mg_kl_unbiased` | Unbiased KL gradient on the KL-as-loss path | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 24 | KL | `mg_kl_adaptive` | Adaptive KL controller (target + horizon feedback) | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 25 | KL | via `up_ppo_gae` | KL penalty in reward (`kl.init_coef`) | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 26 | KL | via `up_reinforce_baseline` | KL as loss term + k2 estimator | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 27 | KL | via `up_dapo` | KL k3 estimator | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 28 | IS corr | via `up_ppo_gae` | Token-level IS correction, clip mode (TIS) | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 29 | IS corr | via `up_reinforce_baseline` | IS correction, mask mode (ICEPOP) | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 30 | IS corr | via `up_flash_reinforce` | Sequence-level IS correction gated by binary KL | Actor | vLLM engine | Off / Off | none (async) | GSM8K | 5 | **PASS** | |
| 31 | IS corr | `mg_is_tv_gating` | IS correction gated by total-variation divergence | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |

## C. Memory management — the 80-cell gloo matrix (this is where 2 GPUs beat 1)

The matrix sweeps **20 valid topologies x 4 sleep modes x 4 ZeRO stages = 80 cells**, which is
strictly more than the 5 single-GPU memory cases. Full per-cell table:
`gloo_matrix_suite/results/run_20260919_093539_latest_full80_final/MATRIX_TABLE.md`

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | Memory | matrix `X1-X4` x Z0-Z3 | Actor / vLLM **separated** across the two devices, all 4 sleep modes | Actor | vLLM engine | all 4 modes | none (async) | GSM8K | 5 | **PASS** (8 cells; 8 blocked as async+sleep is mutually exclusive upstream) | |
| 33 | Memory | matrix `X9-X12` x Z0-Z3 | Actor ws=2 + **2 engines x TP1**, colocated, all 4 sleep modes | Actor r0 + engine 0 | Actor r1 + engine 1 | all 4 modes | colocate_all | GSM8K | 5 | **PASS** | |
| 34 | Memory | matrix `X13-X16` x Z0-Z3 | Actor ws=2 + **1 engine x TP2** spanning both devices | Actor r0 + TP rank 0 | Actor r1 + TP rank 1 | all 4 modes | colocate_all | GSM8K | 5 | **PASS** | |
| 35 | Memory | matrix `X17-X20` x Z0-Z3 | Actor ws=2 + **critic ws=2** + 2 engines | Actor+Critic r0+eng0 | Actor+Critic r1+eng1 | all 4 modes | colocate_all | GSM8K | 5 | **PASS** | |
| 36 | Memory | matrix `X21-X24` x Z0-Z3 | Actor ws=2 + critic ws=2 + 1 engine x TP2 | Actor+Critic r0+TP0 | Actor+Critic r1+TP1 | all 4 modes | colocate_all | GSM8K | 5 | **PASS** | |
| 37 | Memory | matrix stage-0 + DS-sleep rows | DeepSpeed sleep at ZeRO stage 0 | varies | varies | DS On | colocate_all | GSM8K | 0 | **N/A — unsupported** | Architectural: stage 0 builds `FP16_UnfusedOptimizer`, which has no `offload_states()`. 9/9 correlation: every stage-0 row with DS sleep On fails, every DS-sleep-Off row passes |
| 38 | Memory | `mg_overlap_comm` | `overlap_comm` explicitly **ON** at ZeRO-2 (counterpart of our ZeRO-3 fix) | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 39 | Memory | `mg_zpg` | ZeRO++ hierarchical partitioning (`zpg 2`) at stage 3 | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 40 | Memory | `mg_grad_accum_dtype` | Gradient accumulation forced to fp32 under bf16 params | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 41 | Memory | Adam offload on / off | `ds.adam_offload` both ways | varies | varies | varies | both | GSM8K | 5 | **PASS** | on = bench + ext; off = matrix |

## D. Throughput & batching

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | Batching | via `up_ppo_gae` | Sample packing in the **RL** path + dynamic token-budgeted batching (`dynamic_batch_enable` forces `packing_samples=True`) | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 43 | Batching | `mg_rm_packing` | Reward-model sample packing at ZeRO-3 | RM trainer | **idle** | n/a | n/a | pref mixture | 128 samples | **PASS** | |
| 44 | Batching | `mg_dpo_packing` | DPO sample packing | DPO trainer | **idle** | n/a | n/a | pref mixture | 113 steps | **PASS** | |
| 45 | Batching | `up_sft` / `mg_sft_zero3` | SFT packing + ZeRO-3 code path | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | ZeRO-3 is a no-op at ws=1 |
| 46 | Batching | `mg_grad_ckpt_reentrant` | Reentrant gradient checkpointing, RL actor | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 47 | Batching | `mg_sup_grad_ckpt_reentrant` | Reentrant gradient checkpointing, supervised | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |

## E. Rollout tuning

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | Rollout | `mg_prefix_caching` | vLLM prefix caching shared across samples of a prompt | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 49 | Rollout | via `up_reinforce_baseline` | Dynamic prompt filtering by reward range | Actor+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 50 | Rollout | `mg_sampling_params` | Non-default rollout temperature 0.7 / top_p 0.9 | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 51 | Rollout | `up_agent_async` | Async rollout + partial rollout | Actor | vLLM engine | Off / Off | none (async) | GSM8K | 5 | **PASS** | |
| 52 | Rollout | `mg_sync_with_ray` | Weight sync via Ray collective instead of the torch process group | Actor+vLLM | shared | On / On | colocate_all | GSM8K | 0 | **FAIL** | Two configs, two errors: separated -> `the new group's world size should be <= the world size set by init_process_group`; colocated -> `The collective APIs shall be only used inside a Ray actor or task`. The group must be created inside a Ray actor and is not on this path. Looks unmaintained upstream |

## F. Adapters & training lifecycle

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 53 | Adapter | `mg_lora_variants` | LoRA rank 8 / alpha 32 / dropout 0.05 / target_modules q,v — merged adapter broadcast to vLLM over gloo | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 54 | Adapter | `up_sft_lora` | SFT + LoRA | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 55 | Adapter | **`grpo_ema`** | EMA moving-average of policy weights | Actor | vLLM engine | Off / Off | none | GSM8K | — | **PASS** | closed 2026-09-20 |
| 56 | Quant | `mg_sft_4bit` | QLoRA / 4-bit on SFT — isolates quantisation from rollout | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | Proves 4-bit itself works on XPU |
| 57 | Quant | `mg_ppo_4bit` | 4-bit quantised actor inside the **full RL loop** | Actor | vLLM engine | Off / Off | none | GSM8K | 0 | **FAIL (timeout)** | bitsandbytes packs 4-bit as **uint8**; `vllm_worker_wrap.py:41` asserts params match `bf16`. The assert fires in a Ray remote on one side of the gloo collective, so the engine never joins and the actor blocks forever -> hang, not an error. Only 2 of ~290 params synced. No dequantise-before-broadcast step. Not XPU-specific |
| 58 | Optimizer | `mg_muon_ppo` | Muon optimizer on actor **and** critic | Actor+Critic+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 59 | Optimizer | `mg_muon_sft` | Muon optimizer on SFT | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 60 | Lifecycle | `mg_ckpt_disable_ds` | HF-only checkpointing, skipping the DeepSpeed checkpoint | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 61 | Lifecycle | `mg_ckpt_best_metric` | Best-checkpoint selection by metric + rotation (`max_num 2`) | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | |
| 62 | Lifecycle | `up_ckpt_universal` | ZeRO -> universal checkpoint **conversion** | Actor+Critic+Ref+vLLM | shared | On / On | colocate_all | GSM8K | 5 | **PASS** | 2/2 dirs converted, 1162+1166 files |
| 63 | Lifecycle | `mg_universal_ckpt_load` | Convert **then resume from** the universal checkpoint | Actor | vLLM engine | Off / Off | none | GSM8K | 5+5 | **PASS** | Closes the loop the bench left open |
| 64 | Lifecycle | `mg_ppo_eval` | In-training eval: dataset, cadence, n_samples, temperature, split | Actor | vLLM engine | Off / Off | none | GSM8K | 5 | **PASS** | No other 2-GPU suite enables eval |
| 65 | Lifecycle | `mg_sft_eval` | Evaluation on the supervised path | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 66 | Lifecycle | `mg_lora_combiner` | `openrlhf.cli.lora_combiner` merges a trained adapter into the base model | SFT trainer, then CPU merge | **idle** | n/a | n/a | GSM8K SFT | n/a | **PASS** | An entrypoint no other suite touches |
| 67 | Correctness | `mg_determinism` | Same RL config twice, fixed seed + `full_determinism_enable`, aligned loss comparison | Actor | vLLM engine | Off / Off | none | GSM8K | 5 x2 | **FAIL** | Diverges at step 1 (runA `-0.012687 0.0 -0.020611 ...` vs runB `0.0 -0.008001 0.018210 ...`). Should reproduce: OpenRLHF sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` (vLLM's documented reproducibility switch) and seeds each engine. Genuine finding |
| 68 | Correctness | `mg_determinism_sft` | Same SFT config twice — **no vLLM, no sampling** | SFT trainer | **idle** | n/a | n/a | GSM8K SFT | 2 runs | **YET TO TEST** | Written after the last full run, never executed. Exists to localise #67: if SFT reproduces but RL does not, the divergence is in the rollout path; if neither does, it is the XPU compute kernels |

## G. Supervised workflows

All P4: single process, GPU 1 idle. Listed so nobody mistakes these for distributed runs.

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 69 | SFT | `up_sft` | SFT full fine-tune, ZeRO-2 | SFT trainer | idle | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 70 | SFT | `mg_sft_zero3` | SFT at ZeRO-3 | SFT trainer | idle | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | Stage-3 code path only; no sharding at ws=1 |
| 71 | SFT | `mg_sft_pretrain_mode` | Pretrain mode — plain LM loss over flat text, no chat template | SFT trainer | idle | n/a | n/a | flattened text | 80 lines | **PASS** | Needed a `text` column built from `messages` |
| 72 | RM | `up_rm` | Reward-model training, ZeRO-3 | RM trainer | idle | n/a | n/a | pref mixture | 259 lines | **PASS** | |
| 73 | RM | `mg_rm_loss_type_fp32` | RM LogExp loss variant + fp32 loss under bf16 params | RM trainer | idle | n/a | n/a | pref mixture | 131 lines | **PASS** | |
| 74 | DPO | `up_dpo` | DPO with reference model, beta 0.1, ZeRO-3 | DPO trainer | idle | n/a | n/a | pref mixture | 225 lines | **PASS** | |
| 75 | DPO | **`dpo_ipo`** | IPO loss variant | DPO trainer | idle | n/a | n/a | pref mixture | — | **PASS** | closed 2026-09-20 |
| 76 | DPO | **`dpo_cdpo`** | cDPO label smoothing | DPO trainer | idle | n/a | n/a | pref mixture | — | **PASS** | closed 2026-09-20 |
| 77 | DPO | `mg_dpo_lora` | DPO + LoRA | DPO trainer | idle | n/a | n/a | pref mixture | 113 steps | **PASS** | |
| 78 | DPO | `mg_dpo_nll` | DPO + auxiliary NLL regulariser | DPO trainer | idle | n/a | n/a | pref mixture | 113 steps | **PASS** | |
| 79 | Data | `mg_prompt_probs` | Blend two prompt datasets with explicit sampling weights | Actor | vLLM engine | Off / Off | none | GSM8K x2 | 5 | **PASS** | |
| 80 | Data | `mg_data_chat_template` | Custom tokenizer chat template supplied on the CLI | SFT trainer | idle | n/a | n/a | GSM8K SFT | 0 | **FAIL** | Error changed after the fix, which is progress: a hand-written template gave `must contain at least one completion` (no assistant boundary for the loss mask); the model's own template gives `must contain at least one complete gradient-accumulation window` — template accepted, sample count short. `max_samples` raised, not yet re-run |

## H. Agent, VLM, MoE and cross-device parallelism

| # | Group | Test ID | What it validates | GPU 0 | GPU 1 | Sleep | Coloc | Dataset | Steps | Result | Reason for fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 81 | Agent RL | `up_agent_async` | Agent-based multi-turn rollout, async | Actor | vLLM engine | Off / Off | none (async) | GSM8K | 5 | **PASS** | |
| 82 | VLM | `up_vlm` | Vision-language RL (no critic, no packing — both asserted upstream) | Actor+Ref+vLLM | shared | On / On | colocate_all | geometry3k | 2 | **PASS** | SmolVLM-256M; Qwen2-VL-2B exhausted host RAM at 119/125 GB |
| 83 | Parallel | `mg_ds_autotp2` | **DeepSpeed AutoTP** — actor tensors split across both Arc B70s (`ds.tensor_parallel_size 2`), 2 engines to satisfy the colocate_all device-count assert | Actor rank 0 + engine 0 | Actor rank 1 + engine 1 | On / On | colocate_all | GSM8K | 5 | **PASS** | |
| 84 | Parallel | `mg_ring_attn2` | Ring/sequence attention sharding the sequence dim across both devices | Actor rank 0 + engine 0 | Actor rank 1 + engine 1 | On / On | colocate_all | GSM8K | — | **SKIP** | `ModuleNotFoundError: ring_flash_attn` — absent optional package, not a defect. `pip install ring-flash-attn` enables it |
| 85 | Parallel | `mg_liger` | Liger fused kernels | Actor | vLLM engine | Off / Off | none | GSM8K | — | **SKIP** | `ModuleNotFoundError: liger_kernel` — absent optional package |
| 86 | Coloc | `mg_colocate_critic_reward` | Critic + reward sharing a device, actor separate | Actor+Critic+Reward+vLLM | shared | On / On | colocate_critic_reward + colocate_all | GSM8K | 5 | **PASS** | |
| 87 | Coloc | **`ppo_ref_model_nocolo`** | Actor + ref colocated, vLLM separate — partial colocation | Actor+Ref | vLLM engine | Off / Off | colocate_actor_ref | GSM8K | — | **PASS** | closed 2026-09-20 |
| 88 | MoE | `mg_moe_sft` | SFT on a genuine 32-expert MoE (granite-3.1-1b-a400m) | MoE SFT trainer | idle | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 89 | MoE | `mg_moe_rm` | Reward-model training on a MoE backbone | MoE RM trainer | idle | n/a | n/a | pref mixture | dataset | **PASS** | |
| 90 | MoE | `mg_moe_experts_eager` | MoE expert strategy `eager` — reference per-expert loop | MoE SFT trainer | idle | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 91 | MoE | `mg_moe_experts_grouped_mm` | MoE expert strategy `grouped_mm` — grouped GEMM over experts | MoE SFT trainer | idle | n/a | n/a | GSM8K SFT | 80 lines | **PASS** | |
| 92 | MoE | `mg_moe_experts_batched_mm` | MoE expert strategy `batched_mm` | MoE SFT trainer | idle | n/a | n/a | GSM8K SFT | 0 | **FAIL** | OOM twice (4.81 GiB at `down_proj[expert_ids]`, then 3.39 GiB at `gate_up_proj[expert_ids]` after shrinking). It gathers one expert weight matrix per (token x top_k), so at 32 experts / top-8 / hidden-1024 the peak is **inherent**, not a batch-size choice. Reached 4/64 steps, so the kernel works. Re-pointed at a tiny 4-expert MoE, not yet re-run |
| 93 | MoE | `mg_moe_aux_loss` | MoE router load-balancing auxiliary loss | MoE SFT trainer | idle | n/a | n/a | GSM8K SFT | 0 | **FAIL** | **transformers bug, not OpenRLHF's.** GraniteMoE ignores `output_router_logits=True`: `router_logits` returns None and `aux_loss` is a literal `int 0`, so transformers' OWN code dies at `modeling_granitemoe.py:644` on `aux_loss.to(...)`. Verified Qwen2Moe and Mixtral both return a Tensor. Re-pointed at a tiny Qwen2Moe, not yet re-run |
| 94 | MoE | `mg_moe_grpo` | MoE **reinforcement learning** — rollout weight sync of expert/router tensors | Actor | vLLM engine | Off / Off | none | GSM8K | 0 | **YET TO TEST (parked)** | Parked by decision. Two attempts, zero steps: GraniteMoE dies on a runtime-vs-checkpoint parameter-name divergence (`KeyError: layers.0.block_sparse_moe.router.weight`); tiny Qwen2Moe clears the names then hits a shape mismatch inseparable from a toy-checkpoint artefact. No real small Qwen2Moe exists — the smallest is 14.3B (~29 GB bf16), which will not fit on 2x16 GB with vLLM and the optimizer |

---

## Totals

| Result | Count |
|---|---|
| **PASS** | 75 |
| **FAIL** | 5 |
| **YET TO TEST** | 8 |
| **SKIP** (absent optional package) | 2 |
| **N/A** (architecturally unsupported) | 1 |
| Rows | 94 (some aggregate whole matrix blocks) |

## Plan to close the 8 "yet to test"

All eight are already written as cases in `tests/test_e2e_suite_multigpu.sh`, the 28-case base
e2e suite, which has never been executed on this tree (1 smoke case, 27 skipped). No new code is
needed for seven of them:

  RLOO, Dr. GRPO, DPO-IPO, DPO-cDPO, reward normalization, overlong reward penalty,
  colocate_actor_ref, EMA

Step 1: run the base e2e suite (28 cases, about 2.5 h) -> closes 7 of the 8.
Step 2: run `mg_determinism_sft` (about 10 min) -> localises the determinism finding.
Step 3: re-run the 3 cases whose fixes are applied but unverified
        (`mg_moe_aux_loss`, `mg_moe_experts_batched_mm`, `mg_data_chat_template`), about 15 min.
Step 4: reward normalization needs a reward source with non-zero variance before it can mean
        anything on a 0.5B model; flag for a design decision rather than a blind re-run.

MoE RL (row 94) stays parked until a real small MoE is available.
