# 2-GPU coverage: every testable behaviour

- Repo under test: `/home/sdp/madhu/OpenRLHF-fresh` (latest upstream + our changes)
- Proven sources: gloo matrix **63 PASS**, upstream bench **14 PASS**
- e2e suite exists but 27 of its 28 cases have NEVER run on this tree

**PASS 74 | RE-RUN 8 | TO TEST 51 | total 133**

| # | Group | Use case | Flag / value | Evidence | Status |
|---|---|---|---|---|---|
| 1 | A Advantage estimator | GAE (PPO + critic) | `estimator gae` | bench | **PASS** |
| 2 | A Advantage estimator | REINFORCE | `estimator reinforce` | bench | **PASS** |
| 3 | A Advantage estimator | REINFORCE++ baseline | `estimator reinforce_baseline` | bench | **PASS** |
| 4 | A Advantage estimator | RLOO | `estimator rloo` | e2e (never run on latest tree) | **RE-RUN** |
| 5 | A Advantage estimator | GRPO / group_norm | `estimator group_norm` | bench | **PASS** |
| 6 | A Advantage estimator | DR-GRPO | `estimator dr_grpo` | e2e (never run on latest tree) | **RE-RUN** |
| 7 | A Advantage estimator | FlashREINFORCE | `estimator flash_reinforce` | bench | **PASS** |
| 8 | A Advantage estimator | No std-dev normalization | `--algo.advantage.no_std_norm` | - | **TO TEST** |
| 9 | A Advantage estimator | GAE gamma/lambda non-default | `--algo.advantage.lambd` | - | **TO TEST** |
| 10 | B Policy loss | PPO clipped surrogate | `policy_loss_type ppo` | default in every case | **PASS** |
| 11 | B Policy loss | GSPO | `policy_loss_type gspo` | - | **TO TEST** |
| 12 | B Policy loss | Clip-higher (asymmetric eps) | `--actor.eps_clip_low_high` | bench | **PASS** |
| 13 | B Policy loss | Dual clip | `--actor.dual_clip` | - | **TO TEST** |
| 14 | B Policy loss | token-mean aggregation | `loss_agg_mode token-mean` | default in every case | **PASS** |
| 15 | B Policy loss | seq-mean-token-mean aggregation | `loss_agg_mode seq-mean-token-mean` | bench | **PASS** |
| 16 | B Policy loss | Entropy bonus | `--actor.entropy_coef` | bench | **PASS** |
| 17 | B Policy loss | MoE aux loss | `--actor.aux_loss_coef` | - | **TO TEST** |
| 18 | C KL control | KL penalty in reward (init_coef>0) | `--algo.kl.init_coef` | matrix+bench | **PASS** |
| 19 | C KL control | KL as loss term | `--algo.kl.use_loss` | bench | **PASS** |
| 20 | C KL control | k1 estimator | `kl.estimator k1` | default in every case | **PASS** |
| 21 | C KL control | k2 estimator | `kl.estimator k2` | bench | **PASS** |
| 22 | C KL control | k3 estimator | `kl.estimator k3` | bench | **PASS** |
| 23 | C KL control | Unbiased KL gradient | `--algo.kl.unbiased_gradient` | - | **TO TEST** |
| 24 | C KL control | Adaptive KL (target/horizon) | `--algo.kl.target` | - | **TO TEST** |
| 25 | D IS correction | Off (default) | `is_correction_level off` | default in every case | **PASS** |
| 26 | D IS correction | Token level | `is_correction_level token` | bench | **PASS** |
| 27 | D IS correction | Sequence level | `is_correction_level seq` | bench | **PASS** |
| 28 | D IS correction | mask mode (ICEPOP) | `is_correction_mode mask` | bench | **PASS** |
| 29 | D IS correction | clip mode (TIS) | `is_correction_mode clip` | bench | **PASS** |
| 30 | D IS correction | ratio gating | `is_correction_gating ratio` | default in every case | **PASS** |
| 31 | D IS correction | binary_kl gating | `is_correction_gating binary_kl` | bench | **PASS** |
| 32 | D IS correction | tv gating | `is_correction_gating tv` | - | **TO TEST** |
| 33 | E Reward/reference | Remote reward function | `--reward.remote_url` | matrix+bench | **PASS** |
| 34 | E Reward/reference | Reward MODEL (4th model) | `--reward.model_name_or_path` | bench | **PASS** |
| 35 | E Reward/reference | Reward normalization | `--reward.normalize_enable` | bench | **PASS** |
| 36 | E Reward/reference | Reward clip range | `--reward.clip_range` | - | **TO TEST** |
| 37 | E Reward/reference | Overlong reward penalty | `--reward.overlong_buffer_len` | e2e (never run on latest tree) | **RE-RUN** |
| 38 | E Reward/reference | Stop-properly penalty | `--reward.stop_properly_penalty_coef` | bench | **PASS** |
| 39 | E Reward/reference | Reward-model CPU offload | `--reward.offload` | - | **TO TEST** |
| 40 | E Reward/reference | Reference-model CPU offload | `--ref.offload` | e2e (never run on latest tree) | **RE-RUN** |
| 41 | F Placement | colocate_all | `--train.colocate_all` | matrix+bench | **PASS** |
| 42 | F Placement | colocate_actor_ref | `--train.colocate_actor_ref` | bench | **PASS** |
| 43 | F Placement | colocate_critic_reward | `--train.colocate_critic_reward` | - | **TO TEST** |
| 44 | F Placement | No colocation (roles separated) | `num_gpus_per_node 1` | matrix scenarios X1-X4 | **PASS** |
| 45 | F Placement | Async rollout | `--train.async_enable` | matrix+bench | **PASS** |
| 46 | F Placement | Partial rollout | `--train.partial_rollout_enable` | bench | **PASS** |
| 47 | G vLLM | 1 engine x TP1 | `tensor_parallel_size 1` | bench | **PASS** |
| 48 | G vLLM | 2 engines x TP1 | `num_engines 2` | matrix scenarios X9-X12, X17-X20 | **PASS** |
| 49 | G vLLM | 1 engine x TP2 (spans both XPUs) | `tensor_parallel_size 2` | matrix scenarios X13-X16, X21-X24 | **PASS** |
| 50 | G vLLM | vLLM sleep on | `--vllm.enable_sleep` | matrix+bench | **PASS** |
| 51 | G vLLM | enforce_eager | `--vllm.enforce_eager` | matrix+bench | **PASS** |
| 52 | G vLLM | Prefix caching | `--vllm.enable_prefix_caching` | - | **TO TEST** |
| 53 | G vLLM | gloo weight sync | `sync_backend gloo` | bench | **PASS** |
| 54 | G vLLM | Weight sync via Ray | `--vllm.sync_with_ray` | - | **TO TEST** |
| 55 | H DeepSpeed | Stage 0 | `zero_stage 0` | matrix stage sweep: 9 PASS | **PASS** |
| 56 | H DeepSpeed | ZeRO-1 | `zero_stage 1` | matrix stage sweep: 18 PASS | **PASS** |
| 57 | H DeepSpeed | ZeRO-2 | `zero_stage 2` | bench | **PASS** |
| 58 | H DeepSpeed | ZeRO-3 | `zero_stage 3` | bench | **PASS** |
| 59 | H DeepSpeed | DeepSpeed sleep (offload states) | `--ds.enable_sleep` | matrix+bench | **PASS** |
| 60 | H DeepSpeed | Adam CPU offload | `--ds.adam_offload` | matrix+bench | **PASS** |
| 61 | H DeepSpeed | overlap_comm explicit | `--ds.overlap_comm` | - | **TO TEST** |
| 62 | H DeepSpeed | AutoTP tensor parallel = 2 | `--ds.tensor_parallel_size` | - | **TO TEST** |
| 63 | H DeepSpeed | Ring attention size = 2 | `--ds.ring_attn_size` | - | **TO TEST** |
| 64 | H DeepSpeed | Sample packing | `--ds.packing_samples` | bench | **PASS** |
| 65 | H DeepSpeed | 4-bit / QLoRA load | `--ds.load_in_4bit` | - | **TO TEST** |
| 66 | H DeepSpeed | Liger fused kernels | `--ds.use_liger_kernel` | - | **TO TEST** |
| 67 | H DeepSpeed | DeepCompile | `--ds.deepcompile` | - | **TO TEST** |
| 68 | H DeepSpeed | Universal checkpoint load | `--ds.use_universal_ckpt` | - | **TO TEST** |
| 69 | H DeepSpeed | ZeRO++ hierarchical partition (zpg) | `--ds.zpg` | - | **TO TEST** |
| 70 | I Batching | Dynamic token-budgeted batching | `--train.dynamic_batch_enable` | bench | **PASS** |
| 71 | I Batching | Gradient checkpointing | `gradient_checkpointing_enable` | matrix+bench | **PASS** |
| 72 | I Batching | Reentrant gradient checkpointing | `gradient_checkpointing_reentrant` | - | **TO TEST** |
| 73 | I Batching | Rollout max_tokens_per_gpu budget | `--rollout.max_tokens_per_gpu` | bench | **PASS** |
| 74 | J Adapters/optim | LoRA on the actor | `--ds.lora.rank` | matrix+bench | **PASS** |
| 75 | J Adapters/optim | EMA of policy weights | `--train.enable_ema` | e2e (never run on latest tree) | **RE-RUN** |
| 76 | J Adapters/optim | Muon optimizer (actor) | `--actor.optim` | - | **TO TEST** |
| 77 | J Adapters/optim | Muon optimizer (critic) | `--critic.optim` | - | **TO TEST** |
| 78 | J Adapters/optim | Muon optimizer (supervised) | `--optim` | - | **TO TEST** |
| 79 | K Rollout | Dynamic prompt filtering | `--algo.dynamic_filtering_enable` | bench | **PASS** |
| 80 | K Rollout | n_samples_per_prompt > 1 (group sampling) | `n_samples_per_prompt 4` | bench | **PASS** |
| 81 | L Lifecycle | DeepSpeed checkpoint save | `--ckpt.save_steps` | matrix+bench | **PASS** |
| 82 | L Lifecycle | Resume from checkpoint | `--ckpt.load_enable` | bench | **PASS** |
| 83 | L Lifecycle | HF-format export | `--ckpt.save_hf` | bench | **PASS** |
| 84 | L Lifecycle | Skip DeepSpeed ckpt (HF only) | `--ckpt.disable_ds` | - | **TO TEST** |
| 85 | L Lifecycle | ZeRO -> universal conversion | `ds_to_universal` | bench | **PASS** |
| 86 | L Lifecycle | Save critic value network | `--critic.save_value_network` | - | **TO TEST** |
| 87 | L Lifecycle | Critic freezing steps | `--critic.freezing_steps` | - | **TO TEST** |
| 88 | M Evaluation | In-training eval cadence | `--eval.steps` | matrix+bench | **PASS** |
| 89 | M Evaluation | Separate eval dataset | `--eval.dataset` | - | **TO TEST** |
| 90 | M Evaluation | Eval n_samples_per_prompt | `--eval.n_samples_per_prompt` | - | **TO TEST** |
| 91 | M Evaluation | Eval temperature | `--eval.temperature` | - | **TO TEST** |
| 92 | N Correctness | Full determinism | `--train.full_determinism_enable` | - | **TO TEST** |
| 93 | O Agent/VLM | Agent multi-turn rollout | `--train.agent_func_path` | bench | **PASS** |
| 94 | O Agent/VLM | VLM RL with images | `--data.image_key` | bench | **PASS** |
| 95 | O Agent/VLM | Freeze visual encoder | `--actor.freeze_visual_encoder` | bench | **PASS** |
| 96 | P SFT | SFT full fine-tune | `cli.train_sft` | bench up_sft | **PASS** |
| 97 | P SFT | SFT + LoRA | `train_sft` | bench | **PASS** |
| 98 | P SFT | SFT packing | `train_sft` | bench | **PASS** |
| 99 | P SFT | SFT pretrain mode | `--model.pretrain_mode_enable` | - | **TO TEST** |
| 100 | P SFT | SFT multiturn dataset | `--data.multiturn` | - | **TO TEST** |
| 101 | Q RM | Reward-model training | `cli.train_rm` | bench up_rm | **PASS** |
| 102 | Q RM | RM + LoRA | `train_rm` | bench | **PASS** |
| 103 | Q RM | RM packing | `train_rm` | bench | **PASS** |
| 104 | Q RM | RM margin loss | `--model.margin_loss_enable` | - | **TO TEST** |
| 105 | Q RM | RM fp32 loss | `--model.compute_fp32_loss_enable` | - | **TO TEST** |
| 106 | Q RM | RM loss type variant | `--model.loss_type` | - | **TO TEST** |
| 107 | R DPO | DPO with reference model | `cli.train_dpo` | bench up_dpo | **PASS** |
| 108 | R DPO | IPO loss | `ipo` | e2e (never run on latest tree) | **RE-RUN** |
| 109 | R DPO | cDPO label smoothing | `label_smoothing` | e2e (never run on latest tree) | **RE-RUN** |
| 110 | R DPO | DPO + LoRA | `train_dpo` | bench | **PASS** |
| 111 | R DPO | DPO packing | `train_dpo` | bench | **PASS** |
| 112 | R DPO | DPO + NLL aux loss | `--model.nll_loss_coef` | - | **TO TEST** |
| 113 | S Data | Chat template applied | `--data.apply_chat_template` | matrix+bench | **PASS** |
| 114 | S Data | Custom prompt_key | `--data.prompt_key` | - | **TO TEST** |
| 115 | S Data | Custom tokenizer chat template | `--data.tokenizer_chat_template` | - | **TO TEST** |
| 116 | T Utilities | Remote reward-model HTTP server | `cli.serve_rm` | bench | **PASS** |
| 117 | T Utilities | LoRA combiner | `cli.lora_combiner` | - | **TO TEST** |
| 118 | A Advantage estimator | Discount gamma != 1 (reinforce only) | `--algo.advantage.gamma` | bench | **PASS** |
| 119 | A Advantage estimator | GAE lambda != 1 | `--algo.advantage.lambd` | - | **TO TEST** |
| 120 | B Policy loss | Critic value clipping | `--critic.value_clip` | - | **TO TEST** |
| 121 | E Reward/reference | Overlong penalty factor | `--reward.overlong_penalty_factor` | e2e (never run on latest tree) | **RE-RUN** |
| 122 | L Lifecycle | Best-checkpoint selection by metric | `--ckpt.best_metric_key` | - | **TO TEST** |
| 123 | L Lifecycle | Checkpoint rotation (max_num / max_mem) | `--ckpt.max_num` | bench | **PASS** |
| 124 | M Evaluation | Eval dataset split | `--eval.split` | - | **TO TEST** |
| 125 | S Data | Multi-dataset mixing weights | `--data.prompt_probs` | - | **TO TEST** |
| 126 | O Agent/VLM | Multiple images per prompt | `--data.max_images_per_prompt` | bench | **PASS** |
| 127 | J Adapters/optim | LoRA target_modules / alpha / dropout | `--ds.lora.target_modules` | - | **TO TEST** |
| 128 | H DeepSpeed | Grad-accum dtype override | `--ds.grad_accum_dtype` | - | **TO TEST** |
| 129 | H DeepSpeed | MoE experts implementation | `--ds.experts_implementation` | - | **TO TEST** |
| 130 | H DeepSpeed | attn_implementation eager | `attn_implementation eager` | bench | **PASS** |
| 131 | H DeepSpeed | attn_implementation flash_attention_2 | `attn_implementation flash_attention_2` | - | **TO TEST** |
| 132 | K Rollout | Sampling temperature / top_p control | `--rollout.temperature` | bench | **PASS** |
| 133 | N Correctness | Fixed seed reproducibility | `--train.seed` | - | **TO TEST** |
