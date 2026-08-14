#!/usr/bin/env bash
# =============================================================================
# OpenRLHF XPU E2E Test Suite — Multi-GPU (2x Arc Pro B70)
# =============================================================================
#
# Validates every major OpenRLHF training path on 2x Intel Arc Pro B70 (XPU).
# Covers 29 configurations across 9 groups:
#
#   Group 1  — GRPO variants       (ZeRO-2/3, colocation, LoRA, KL, ref model)
#   Group 2  — Other RL algorithms (REINFORCE, REINFORCE++, RLOO, DR-GRPO)
#   Group 3  — Advanced RL recipes (DAPO, ProRL v2, async, multi-turn agent)
#   Group 4  — Reward shaping      (overlong penalty, reward norm, IS correction)
#   Group 5  — EMA checkpoint
#   Group 6  — VLM                 (Qwen2-VL-2B, image+text, frozen vision encoder)
#   Group 7  — SFT                 (full fine-tune, LoRA, sample packing)
#   Group 8  — Reward Model        (full fine-tune, LoRA)
#   Group 9  — DPO / IPO / cDPO
#
# Hardware: 2x Arc Pro B70 (XPU), torch 2.12.0+xpu, vLLM 0.23.1rc1, Ray 2.55
# Model:    Qwen/Qwen2.5-0.5B (RL/SFT/RM/DPO), Qwen2-VL-2B-Instruct (VLM)
# Dataset:  GSM8K prompts (RL), GSM8K SFT (supervised),
#           OpenRLHF preference mixture (RM/DPO), geometry3k (VLM)
#
# NOTE — ppo_gae (PPO with critic) requires actor+critic+vLLM = 3 GPUs minimum
# and is NOT included here. It is validated separately on a 3+ GPU NVIDIA box.
#
# Usage:
#   bash tests/test_e2e_suite.sh            # run all tests
#   bash tests/test_e2e_suite.sh grpo       # run tests whose ID contains "grpo"
#   bash tests/test_e2e_suite.sh sft        # run only SFT tests
#
# Results:
#   tests/results/e2e_suite_<timestamp>/
#     summary.txt   — one line per test: PASS/FAIL/SKIP
#     <id>.log      — full training output
#
# Exit code: 0 if all run tests passed, 1 if any failed.
# =============================================================================

set -uo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REPO=/home/sdp/madhu/OpenRLHF-multi
VENV=/home/sdp/madhu/openrlhf_exp/OpenRLHF/venv-tf

# Default model for all RL/SFT/RM/DPO tests (small, fast, cached locally)
MODEL=Qwen/Qwen2.5-0.5B

# VLM model — downloaded to /tmp to avoid root-owned HF cache locks from Docker
VLM_MODEL=/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct

# Reward function used by all RL tests (math \boxed{} answer extraction)
REWARD_FN=$REPO/examples/python/math_reward_func.py

# Datasets
PROMPTS=$REPO/gsm8k_train_prompts.jsonl              # RL: 80 GSM8K math prompts
SFT_DATA=/home/sdp/data/gsm8k_sft/train.parquet     # SFT: 128 chat-format GSM8K samples
PREF_DATA=OpenRLHF/preference_dataset_mixture2_and_safe_pku  # RM/DPO: preference pairs
VLM_DATA=hiyouga/geometry3k                          # VLM: 32 visual geometry problems

# XPU device environment (Arc Pro B70, Ray misdetects as NVIDIA without this)
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$VENV/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1

# Weight-freshness probe: set to 1 to log actor/vLLM checksums at each sync step
export OPENRLHF_WEIGHT_PROBE=0

# Optional HF dataset cache override (avoids root-owned lock files from Docker)
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite

FILTER="${1:-}"
TS=$(date +%Y%m%dT%H%M%S)
RESULTS=$REPO/tests/results/e2e_suite_$TS
mkdir -p "$RESULTS"
SUMMARY=$RESULTS/summary.txt
PASS=0; FAIL=0; SKIP=0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Stop Ray and kill any leftover training/engine processes.
# Called before every test to guarantee a clean GPU state.
ray_start() {
    local n=${1:-2}
    "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "train_ppo_ray|train_sft|train_dpo|train_rm|EngineCore|PolicyModelActor|RolloutRayActor|ray::" \
        >/dev/null 2>&1 || true
    rm -rf /tmp/ray
    sleep 4
    "$VENV/bin/ray" start --head --num-gpus="$n" 2>&1 | tail -3
    sleep 3
}

ray_stop() {
    "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "train_ppo_ray|train_sft|train_dpo|train_rm|EngineCore|PolicyModelActor|RolloutRayActor|ray::" \
        >/dev/null 2>&1 || true
    rm -rf /tmp/ray
    sleep 4
}

# run_ppo_test ID DESCRIPTION [extra flags...]
#
# Runs a 10-step RL training job via train_ppo_ray with the common base
# configuration (1 actor GPU + 1 vLLM GPU, gloo weight-sync, Qwen2.5-0.5B,
# GSM8K prompts, ZeRO-2 by default). Extra flags override or extend the base.
#
# PASS criteria: exit 0 AND at least 1 "Global step" line in the log.
# Note: DAPO/ProRL v2 may produce fewer than 10 steps because dynamic
# filtering discards batches where all responses score identically — this is
# correct algorithm behaviour, not a failure.
run_ppo_test() {
    local id=$1 desc=$2; shift 2
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"
        echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }

    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    ray_start 2
    "$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
        --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
        --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
        --vllm.gpu_memory_utilization 0.8 --vllm.enforce_eager \
        --vllm.sync_backend gloo \
        --actor.model_name_or_path "$MODEL" \
        --reward.remote_url "$REWARD_FN" \
        --data.prompt_dataset "$PROMPTS" \
        --data.input_key prompt --data.label_key label --data.apply_chat_template \
        --data.max_len 512 --data.max_samples 80 \
        --train.batch_size 8 --train.micro_batch_size 4 \
        --train.max_epochs 1 \
        --ds.attn_implementation sdpa \
        --ckpt.output_dir /tmp/e2e_"$id" --ckpt.save_steps -1 \
        --logger.logging_steps 1 --eval.steps -1 \
        "$@" > "$logf" 2>&1
    local rc=$?
    ray_stop

    local steps
    steps=$(grep -c "Global step" "$logf" 2>/dev/null || echo 0)
    if [[ $rc -eq 0 && $steps -ge 1 ]]; then
        log "PASS  $id ($steps steps)"
        echo "PASS  $id — $desc ($steps steps)" >> "$SUMMARY"
        ((PASS++))
    else
        log "FAIL  $id (exit=$rc steps=$steps)"
        echo "FAIL  $id — $desc (exit=$rc, steps=$steps)" >> "$SUMMARY"
        ((FAIL++))
    fi
}

# run_supervised_test ID DESCRIPTION TRAINER [extra flags...]
#
# Runs a supervised training job (SFT, RM, or DPO) directly via DeepSpeed —
# no Ray cluster, no vLLM. Runs on a single XPU.
#
# PASS criteria: exit 0 AND at least 1 "loss=" line in the log.
run_supervised_test() {
    local id=$1 desc=$2 trainer=$3; shift 3
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"
        echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }

    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    PYTHONUNBUFFERED=1 "$VENV/bin/python" -m openrlhf.cli."$trainer" \
        --ds.zero_stage 2 --ds.adam_offload \
        --ds.attn_implementation sdpa \
        --ds.param_dtype bf16 \
        --train.max_epochs 1 \
        --logger.logging_steps 1 \
        "$@" > "$logf" 2>&1
    local rc=$?

    local loss_lines
    loss_lines=$(grep -cE "loss=|Loss:" "$logf" 2>/dev/null || echo 0)
    if [[ $rc -eq 0 && $loss_lines -ge 1 ]]; then
        log "PASS  $id (loss_lines=$loss_lines)"
        echo "PASS  $id — $desc (loss_lines=$loss_lines)" >> "$SUMMARY"
        ((PASS++))
    else
        log "FAIL  $id (exit=$rc loss_lines=$loss_lines)"
        echo "FAIL  $id — $desc (exit=$rc, loss_lines=$loss_lines)" >> "$SUMMARY"
        ((FAIL++))
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# TEST SUITE
# ──────────────────────────────────────────────────────────────────────────────

log "══════════════════════════════════════════════"
log "OpenRLHF XPU E2E Test Suite  $TS"
log "Hardware: 2x Arc Pro B70 | Model: $MODEL"
log "Results:  $RESULTS"
log "══════════════════════════════════════════════"

# ── Group 1: GRPO variants ────────────────────────────────────────────────────
# GRPO (group_norm) is a group-based reward method — no critic model needed.
# These tests cover the primary no-colocation 2-GPU path (actor on XPU:0,
# vLLM rollout engine on XPU:1, gloo weight-sync every step), as well as
# colocation, LoRA, and KL-regularised variants.

run_ppo_test "grpo_zero2_nocolo" \
    "GRPO, ZeRO-2, no colocation — baseline path" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "grpo_zero3_nocolo" \
    "GRPO, ZeRO-3 + adam_offload — shards model params across GPUs" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 3 --ds.adam_offload

run_ppo_test "grpo_colo_sleep" \
    "GRPO, colocate_all + sleep — actor and vLLM time-share the same GPUs" \
    --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.gpu_memory_utilization 0.4 \
    --train.colocate_all --vllm.enable_sleep --ds.enable_sleep \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "grpo_lora_nocolo" \
    "GRPO, LoRA rank=16 — adapter-only training, base model frozen" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload \
    --ds.lora.rank 16 --ds.lora.alpha 32

run_ppo_test "grpo_kl_kluse" \
    "GRPO + KL as loss (k3, use_loss) — KL added directly to policy loss" \
    --algo.advantage.estimator group_norm \
    --algo.kl.init_coef 0.01 --algo.kl.use_loss --algo.kl.estimator k3 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --train.colocate_actor_ref \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

# Reference model path: actor and reference model colocated on GPU:0, vLLM on GPU:1.
# ppo_gae (PPO with a separate critic) requires 3 roles and is not testable on
# this 2-GPU box. These two tests exercise the reference-model code path instead.
run_ppo_test "ppo_ref_model_nocolo" \
    "GRPO + reference model KL (colocate_actor_ref) — exercises reference forward path" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0.01 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --train.colocate_actor_ref \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "ppo_ref_zero2_offload" \
    "GRPO + reference model CPU-offloaded — ref model lives in CPU RAM between uses" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0.01 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --ref.offload \
    --train.colocate_actor_ref \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 2: Other RL algorithms ──────────────────────────────────────────────
# All use the same ppo_actor/weight-sync path as GRPO — only the advantage
# estimator differs. Tests confirm each algorithm reaches a training step.

run_ppo_test "reinforce_nocolo" \
    "REINFORCE — simplest policy gradient, reward × log-prob, no baseline" \
    --algo.advantage.estimator reinforce --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 1 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "reinforce_baseline_nocolo" \
    "REINFORCE++ — REINFORCE with per-group mean baseline for variance reduction" \
    --algo.advantage.estimator reinforce_baseline --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "rloo_nocolo" \
    "RLOO — leave-one-out baseline, lower variance than REINFORCE++" \
    --algo.advantage.estimator rloo --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_ppo_test "dr_grpo_nocolo" \
    "DR-GRPO — GRPO without std-deviation normalisation (bias correction)" \
    --algo.advantage.estimator dr_grpo --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 3: Advanced RL recipes ──────────────────────────────────────────────

# DAPO: GRPO + dynamic filtering (discards batches where all responses score
# the same — no learning signal) + asymmetric clip-higher (larger upward policy
# updates). Low step count is expected: filtering leaves few qualifying batches
# on a 32-sample run with a small model.
run_ppo_test "dapo_nocolo" \
    "DAPO — GRPO + dynamic filtering + asymmetric clip-higher (eps_clip_low_high 0.2/0.27)" \
    --algo.advantage.estimator group_norm \
    --algo.kl.init_coef 0.001 --algo.kl.use_loss --algo.kl.estimator k3 \
    --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0.0 1.0 \
    --actor.eps_clip_low_high 0.2 0.27 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --train.colocate_actor_ref \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --data.max_samples 400 \
    --ds.zero_stage 2 --ds.adam_offload

# ProRL v2: REINFORCE++-baseline + clip-higher + dynamic filtering.
# State-of-the-art recipe for prolonged RL on math/reasoning tasks.
run_ppo_test "prorlv2_nocolo" \
    "ProRL v2 — REINFORCE++-baseline + clip-higher + dynamic filtering" \
    --algo.advantage.estimator reinforce_baseline \
    --algo.kl.init_coef 0.0001 --algo.kl.use_loss --algo.kl.estimator k2 \
    --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0.0 1.0 \
    --actor.eps_clip_low_high 0.2 0.27 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --train.colocate_actor_ref \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --data.max_samples 400 \
    --ds.zero_stage 2 --ds.adam_offload

# Async GRPO: generation and backward pass overlap in time, using
# PPOTrainerAsync instead of the synchronous PPOTrainer.
run_ppo_test "grpo_async" \
    "Async GRPO — generation and backward pass overlap for higher GPU utilisation" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --train.async_enable \
    --ds.zero_stage 2 --ds.adam_offload

# Multi-turn agent: the model interacts with an environment over 1-3 steps per
# prompt instead of generating a single response.
run_ppo_test "grpo_agent_multiturn" \
    "GRPO + multi-turn agent — model interacts with environment over 1-3 steps" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --train.agent_func_path examples/python/agent_func.py \
    --data.input_key prompt \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 4: Reward shaping ───────────────────────────────────────────────────

# Overlong penalty: cosine-shaped reward penalty applied to responses that
# exceed a soft token limit, discouraging unnecessary verbosity.
run_ppo_test "grpo_overlong_penalty" \
    "GRPO + overlong penalty — cosine penalty past soft token limit (buffer_len=50)" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --reward.overlong_buffer_len 50 --reward.overlong_penalty_factor 1.0 \
    --ds.zero_stage 2 --ds.adam_offload

# Reward normalisation: standardise reward distribution (mean=0, std=1) before
# computing advantages — stabilises training when reward scale varies.
run_ppo_test "grpo_reward_norm" \
    "GRPO + reward normalisation — standardises reward distribution across the batch" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --reward.normalize_enable \
    --ds.zero_stage 2 --ds.adam_offload

# IS correction: importance sampling corrects for the off-policy gap between
# the vLLM rollout snapshot and the current training policy.
run_ppo_test "grpo_is_correction" \
    "GRPO + IS correction — token-level TIS for off-policy vLLM rollouts" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --algo.advantage.is_correction_enable \
    --algo.advantage.is_correction_type tis \
    --algo.advantage.is_correction_threshold 0.5 5.0 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 5: EMA checkpoint ───────────────────────────────────────────────────
# Maintains an exponential moving average copy of policy weights alongside
# training. EMA model typically generalises better than the last checkpoint.
# The moving_average call in ppo_actor.py was fixed as part of the device-API
# PR: "cuda" string replaced with torch.accelerator.current_accelerator().type.
run_ppo_test "grpo_ema" \
    "GRPO + EMA checkpoint — exponential moving average of policy weights" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --train.enable_ema --train.ema_beta 0.992 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 6: VLM ──────────────────────────────────────────────────────────────
# Vision-Language Model training with image+text input. Uses Qwen2-VL-2B on
# geometry3k (visual math problems). Vision encoder is frozen; only the
# language model is trained.
#
# Prerequisites:
#   - Model must be downloaded to $VLM_MODEL before running:
#       python -c "from huggingface_hub import snapshot_download; \
#                  snapshot_download('Qwen/Qwen2-VL-2B-Instruct', \
#                  local_dir='/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct')"
#   - geometry3k dataset must be accessible (cached via HF_DATASETS_CACHE)
#   - Requires root-free HF cache — set HF_DATASETS_CACHE to a writable dir

[[ -n "$FILTER" && "vlm_grpo" != *"$FILTER"* ]] && {
    log "SKIP vlm_grpo — filtered"
    echo "SKIP  vlm_grpo — filtered" >> "$SUMMARY"
    ((SKIP++))
} || {
    log "START vlm_grpo — VLM GRPO — Qwen2-VL-2B, image+text, frozen vision encoder"
    logf=$RESULTS/vlm_grpo.log
    ray_start 2
    "$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
        --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
        --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
        --vllm.gpu_memory_utilization 0.7 --vllm.enforce_eager \
        --vllm.sync_backend gloo \
        --actor.model_name_or_path "$VLM_MODEL" \
        --reward.remote_url "$REWARD_FN" \
        --data.prompt_dataset "$VLM_DATA" \
        --data.input_key problem --data.label_key answer \
        --data.apply_chat_template \
        --data.image_key images --data.max_images_per_prompt 1 \
        --actor.freeze_visual_encoder \
        --algo.advantage.estimator reinforce_baseline --algo.kl.init_coef 0 \
        --rollout.n_samples_per_prompt 4 --rollout.batch_size 4 \
        --train.batch_size 4 --train.micro_batch_size 1 \
        --data.max_len 1024 --data.max_samples 32 \
        --train.max_epochs 1 \
        --ds.zero_stage 2 --ds.adam_offload \
        --ds.attn_implementation sdpa \
        --ckpt.output_dir /tmp/e2e_vlm_grpo --ckpt.save_steps -1 \
        --logger.logging_steps 1 --eval.steps -1 > "$logf" 2>&1
    rc=$?; ray_stop
    steps=$(grep -c "Global step" "$logf" 2>/dev/null || echo 0)
    if [[ $rc -eq 0 && $steps -ge 1 ]]; then
        log "PASS  vlm_grpo ($steps steps)"
        echo "PASS  vlm_grpo — VLM GRPO — Qwen2-VL-2B, image+text, frozen vision encoder ($steps steps)" >> "$SUMMARY"
        ((PASS++))
    else
        log "FAIL  vlm_grpo (exit=$rc steps=$steps)"
        echo "FAIL  vlm_grpo — VLM GRPO — Qwen2-VL-2B, image+text, frozen vision encoder (exit=$rc, steps=$steps)" >> "$SUMMARY"
        ((FAIL++))
    fi
}

# ── Group 7: SFT ──────────────────────────────────────────────────────────────
# Supervised Fine-Tuning — no RL, no vLLM, no Ray. Pure DeepSpeed on 1 GPU.
# Covers full fine-tune, LoRA adapters, and sequence packing.

run_supervised_test "sft_zero2" \
    "SFT, ZeRO-2, full fine-tune — next-token prediction on chat-format data" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/e2e_sft_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sft_lora" \
    "SFT, LoRA rank=16 — adapter-only training, base model frozen" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/e2e_sft_lora --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sft_packing" \
    "SFT + sample packing — multiple short sequences packed into max_len, eliminates padding" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.packing_samples \
    --ckpt.output_dir /tmp/e2e_sft_packing --ckpt.save_steps -1 \
    --eval.steps -1

# ── Group 8: Reward Model ─────────────────────────────────────────────────────
# Trains a scalar reward head on {chosen, rejected} preference pairs.
# Covers full fine-tune and LoRA variants.

run_supervised_test "rm_zero2" \
    "Reward Model, ZeRO-2, full fine-tune — binary cross-entropy on preference pairs" \
    train_rm \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/e2e_rm_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "rm_lora" \
    "Reward Model, LoRA rank=16 — memory-efficient RM training with adapters" \
    train_rm \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/e2e_rm_lora --ckpt.save_steps -1 \
    --eval.steps -1

# ── Group 9: DPO / IPO / cDPO ────────────────────────────────────────────────
# Direct preference optimisation and variants. No separate reward model needed;
# uses a frozen reference copy for KL computation.

run_supervised_test "dpo_zero2" \
    "DPO, ZeRO-2 — directly optimises policy to prefer chosen over rejected" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/e2e_dpo_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "dpo_ipo" \
    "IPO — avoids DPO overoptimisation by regressing to log-ratio targets" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --model.ipo_enable \
    --ckpt.output_dir /tmp/e2e_dpo_ipo --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "dpo_cdpo" \
    "cDPO — label smoothing prevents overfitting to noisy preference labels" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --model.label_smoothing 0.1 \
    --ckpt.output_dir /tmp/e2e_dpo_cdpo --ckpt.save_steps -1 \
    --eval.steps -1

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

echo ""                                              >> "$SUMMARY"
echo "────────────────────────────────────"          >> "$SUMMARY"
echo "TOTAL: $((PASS+FAIL+SKIP)) tests"             >> "$SUMMARY"
echo "  PASS: $PASS"                                 >> "$SUMMARY"
echo "  FAIL: $FAIL"                                 >> "$SUMMARY"
echo "  SKIP: $SKIP (filtered)"                      >> "$SUMMARY"
echo "────────────────────────────────────"          >> "$SUMMARY"

log ""
log "══════════════════════════════════════════════"
log "RESULTS: $PASS passed, $FAIL failed, $SKIP skipped"
log "Summary: $SUMMARY"
log "Logs:    $RESULTS/"
log "══════════════════════════════════════════════"
cat "$SUMMARY"

[[ $FAIL -eq 0 ]]
