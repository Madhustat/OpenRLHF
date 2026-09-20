# Algorithm / method coverage on the current tree

Proven sources: bench 14 PASS, gloo matrix 63 PASS, extended suite 46 PASS.
The base e2e suite defines 28 cases but has NEVER run on this tree, so anything only
it covers is UNPROVEN here.

| Category | Algorithm / method | Status | Evidence |
|---|---|---|---|
| advantage estimator | **PPO with critic (GAE)** | PROVEN | bench:up_ppo_gae |
| advantage estimator | **GRPO (group_norm)** | PROVEN | bench:up_dapo |
| advantage estimator | **REINFORCE** | PROVEN | extended:mg_gamma_reinforce |
| advantage estimator | **REINFORCE++ (reinforce_baseline)** | PROVEN | bench:up_reinforce_baseline |
| advantage estimator | **RLOO** | UNPROVEN - base e2e only (never run on this tree) | - |
| advantage estimator | **Dr. GRPO (dr_grpo)** | UNPROVEN - base e2e only (never run on this tree) | - |
| advantage estimator | **FlashREINFORCE** | PROVEN | bench:up_flash_reinforce |
| group_norm + dyn filter + clip-higher + KL-k3 | **DAPO (recipe)** | PROVEN | bench:up_dapo |
| reinforce_baseline + k2 + clip-higher + dyn filter | **ProRL-v2 (recipe)** | PROVEN | bench:up_prorlv2 |
| alternative policy loss | **GSPO policy loss** | PROVEN | extended:mg_gspo |
| trainer | **SFT** | PROVEN | bench:up_sft |
| trainer | **Reward Model (RM)** | PROVEN | bench:up_rm |
| trainer | **DPO** | PROVEN | bench:up_dpo |
| DPO loss variant | **DPO - IPO variant** | UNPROVEN - base e2e only (never run on this tree) | - |
| DPO loss variant | **DPO - cDPO label smoothing** | UNPROVEN - base e2e only (never run on this tree) | - |
| DPO loss variant | **DPO - NLL aux loss** | PROVEN | extended:mg_dpo_nll |
| RM loss variant | **RM - LogExp loss variant** | PROVEN | extended:mg_rm_loss_type_fp32 |
| adapter | **LoRA** | PROVEN | bench:up_sft_lora |
| quantisation | **QLoRA / 4-bit** | PROVEN | extended:mg_sft_4bit |
| optimizer | **Muon optimizer** | PROVEN | extended:mg_muon_ppo |
| reward source | **Remote reward function** | PROVEN | bench:up_ppo_gae |
| reward source | **Reward MODEL (in-loop)** | PROVEN | extended:mg_reward_model |
| reward source | **Reward model HTTP server** | PROVEN | bench:up_serve_rm |
| reward shaping | **Reward normalization** | UNPROVEN - base e2e only (never run on this tree) | - |
| reward shaping | **Reward clip range** | PROVEN | extended:mg_reward_clip_range |
| reward shaping | **Overlong reward penalty** | UNPROVEN - base e2e only (never run on this tree) | - |
| offload | **Reward model CPU offload** | PROVEN | extended:mg_reward_offload |
| offload | **Reference model CPU offload** | PROVEN | extended:mg_ref_offload |
| KL control | **KL penalty in reward** | PROVEN | bench:up_ppo_gae |
| KL control | **KL as loss term** | PROVEN | bench:up_reinforce_baseline |
| KL control | **KL estimator k2** | PROVEN | bench:up_reinforce_baseline |
| KL control | **KL estimator k3** | PROVEN | bench:up_dapo |
| KL control | **Unbiased KL gradient** | PROVEN | extended:mg_kl_unbiased |
| KL control | **Adaptive KL (target/horizon)** | PROVEN | extended:mg_kl_adaptive |
| memory | **vLLM sleep** | PROVEN | bench:up_vlm |
| memory | **DeepSpeed sleep** | PROVEN | bench:up_vlm |
| memory | **Adam CPU offload** | PROVEN | bench:up_agent_async |
| colocation | **colocate_all** | PROVEN | bench:up_vlm |
| colocation | **colocate_actor_ref** | UNPROVEN - base e2e only (never run on this tree) | - |
| colocation | **colocate_critic_reward** | PROVEN | extended:mg_colocate_critic_reward |
| throughput | **Sample packing** | PROVEN | extended:mg_rm_packing |
| throughput | **Dynamic batching** | PROVEN | bench:up_ppo_gae |
| throughput | **Gradient checkpointing** | PROVEN | bench:up_ppo_gae |
| rollout | **Dynamic prompt filtering** | PROVEN | bench:up_reinforce_baseline |
| workflow | **Agent multi-turn rollout** | PROVEN | bench:up_agent_async |
| workflow | **VLM / multimodal RL** | PROVEN | bench:up_vlm |
| workflow | **Async rollout** | PROVEN | bench:up_agent_async |
| workflow | **Partial rollout** | PROVEN | bench:up_agent_async |
| off-policy correction | **IS correction (token/TIS)** | PROVEN | bench:up_ppo_gae |
| off-policy correction | **IS correction (ICEPOP/mask)** | PROVEN | bench:up_reinforce_baseline |
| off-policy correction | **IS correction (seq/binary_kl)** | PROVEN | bench:up_flash_reinforce |
| off-policy correction | **IS correction (tv gating)** | PROVEN | extended:mg_is_tv_gating |
| training lifecycle | **EMA of policy weights** | UNPROVEN - base e2e only (never run on this tree) | - |

## Totals

- PROVEN: 45
- UNPROVEN: 8
