#!/usr/bin/env bash
# =============================================================================
# OpenRLHF XPU E2E Test Suite — Single GPU
# =============================================================================
#
# Validates OpenRLHF training on a SINGLE physical GPU. Every RL test uses
# colocate_all + enable_sleep so the actor and vLLM engine time-share the
# one available GPU. Supervised tests (SFT/RM/DPO) already use 1 GPU natively.
#
# Covers 18 configurations across 6 groups:
#
#   Group 1  — GRPO (colocate_all + sleep)    — baseline, LoRA, KL penalty
#   Group 2  — Other RL algorithms            — REINFORCE, REINFORCE++, RLOO, DR-GRPO
#   Group 3  — Reward shaping                 — overlong penalty, reward norm, IS correction
#   Group 4  — EMA checkpoint
#   Group 5  — SFT                            — full fine-tune, LoRA, sample packing
#   Group 6  — Reward Model + DPO / IPO / cDPO
#
# Async GRPO is NOT included here — it requires >=2 GPUs (see multi-GPU suite).
#
# Hardware: 1 physical GPU (tested on Arc Pro B70 XPU, also valid on NVIDIA)
# Model:    Qwen/Qwen2.5-0.5B
# Dataset:  GSM8K prompts (RL), GSM8K SFT (supervised),
#           OpenRLHF preference mixture (RM/DPO)
#
# Tests NOT included here (require 2+ GPUs):
#   - grpo_zero3_nocolo      — ZeRO-3 no-colocation (actor + vLLM on separate GPUs)
#   - ppo_ref_model_nocolo   — reference model on GPU:0, vLLM on GPU:1
#   - ppo_ref_zero2_offload  — same
#   - dapo_nocolo            — KL ref model needs colocate_actor_ref + vLLM = tight fit
#   - prorlv2_nocolo         — same
#   - grpo_agent_multiturn   — works on 1 GPU but needs venv memory headroom; in multigpu suite
#   - vlm_grpo               — Qwen2-VL-2B requires >1 GPU to fit actor + vLLM
#   - ppo_gae                — requires actor + critic + vLLM = 3 roles minimum
#
# Usage:
#   bash tests/test_e2e_suite_singlegpu.sh            # run all 18 tests
#   bash tests/test_e2e_suite_singlegpu.sh grpo        # run tests whose ID contains "grpo"
#   bash tests/test_e2e_suite_singlegpu.sh sft         # run only SFT tests
#
# Results:
#   tests/results/singlegpu_<timestamp>/
#     summary.txt   — one line per test: PASS/FAIL/SKIP
#     <id>.log      — full training output
#
# Exit code: 0 if all run tests passed, 1 if any failed.
# =============================================================================

set -uo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# LOCAL ADAPTATIONS for this single-XPU box (dut7054) — 2026-08
# ──────────────────────────────────────────────────────────────────────────────
# This suite was authored on a different box (user sdp). The changes below adapt
# it to this machine. Result after adaptation: 18/18 PASS. None of the original
# failures were code defects — all were environment/config/architecture:
#
#   1. REPO / VENV repointed to this box's checkout + venv-tf (paths below).
#   2. PYTHONPATH prepends REPO — the venv's editable openrlhf points at a
#      different checkout, so this forces imports to resolve to THIS repo.
#   3. SFT_DATA repointed to /tmp/gsm8k_sft/train.parquet (the sdp path does not
#      exist here). See PREREQUISITES below to (re)generate it.
#   4. sg_grpo_kl: added `--ref.num_nodes 1 --ref.num_gpus_per_node 1` so the
#      colocated reference model matches the actor's single GPU (otherwise the
#      ref model defaults to 8 GPUs and trips a colocate assertion). CONFIG, not code.
#   5. sample-packing (sg_sft_packing) needs the `kernels` pip package pinned to
#      0.15.2 <= v < 0.16.0. This box had 0.16.0; downgrade once:
#          pip install "kernels==0.15.2"
#      DEPENDENCY VERSION, not code.
#   6. Async GRPO REMOVED from this suite — it requires >=2 GPUs (async overlaps
#      vLLM generation with actor training on separate devices; async+sleep is
#      disallowed and async+colocate-no-sleep deadlocks on one device). It is
#      validated as `grpo_async` in test_e2e_suite_multigpu.sh. ARCHITECTURE, not code.
#
# PREREQUISITES (run once before the suite; both use the venv above):
#   # RL prompts -> $PROMPTS
#   python - <<'PY'
#   from datasets import load_dataset; import json
#   ds = load_dataset("openai/gsm8k","main",split="train").select(range(400))
#   sfx=" Please reason step by step, and put your final answer within \\boxed{}."
#   with open("gsm8k_train_prompts.jsonl","w") as f:
#       for r in ds: f.write(json.dumps({"prompt":r["question"]+sfx,
#           "label":r["answer"].split("####")[-1].strip().replace(",","")})+"\n")
#   PY
#   # SFT parquet ('messages' chat format) -> $SFT_DATA
#   mkdir -p /tmp/gsm8k_sft
#   python - <<'PY'
#   from datasets import load_dataset; import pandas as pd
#   ds = load_dataset("openai/gsm8k","main",split="train").select(range(256))
#   rows=[{"messages":[{"role":"user","content":r["question"]},
#          {"role":"assistant","content":r["answer"]}]} for r in ds]
#   pd.DataFrame(rows).to_parquet("/tmp/gsm8k_sft/train.parquet")
#   PY
#   pip install "kernels==0.15.2"   # required for sg_sft_packing
#
# (RM/DPO download PREF_DATA from HuggingFace on first run — needs network.)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REPO=/home/dut7054/madhu/OpenRLHF_exp_multi
VENV=/home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf
MODEL=Qwen/Qwen2.5-0.5B
REWARD_FN=$REPO/examples/python/math_reward_func.py
PROMPTS=$REPO/gsm8k_train_prompts.jsonl
SFT_DATA=/tmp/gsm8k_sft/train.parquet
PREF_DATA=OpenRLHF/preference_dataset_mixture2_and_safe_pku

# Force the interpreter to use THIS checkout's openrlhf (the venv's editable
# install points at a different checkout; prepend REPO so imports resolve here).
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# XPU device environment — use only the first device for single-GPU runs
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$VENV/bin:$PATH"
export ONEAPI_DEVICE_SELECTOR=level_zero:0     # pin to device 0 only
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_suite

FILTER="${1:-}"
TS=$(date +%Y%m%dT%H%M%S)
RESULTS=$REPO/tests/results/singlegpu_$TS
mkdir -p "$RESULTS"
SUMMARY=$RESULTS/summary.txt
PASS=0; FAIL=0; SKIP=0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Stop Ray and kill any leftover processes before every test.
ray_start() {
    "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "train_ppo_ray|train_sft|train_dpo|train_rm|EngineCore|PolicyModelActor|RolloutRayActor|ray::" \
        >/dev/null 2>&1 || true
    rm -rf /tmp/ray
    sleep 4
    # Single GPU: start Ray with 1 GPU
    "$VENV/bin/ray" start --head --num-gpus=1 2>&1 | tail -3
    sleep 3
}

ray_stop() {
    "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "train_ppo_ray|train_sft|train_dpo|train_rm|EngineCore|PolicyModelActor|RolloutRayActor|ray::" \
        >/dev/null 2>&1 || true
    rm -rf /tmp/ray
    sleep 4
}

# run_rl_singlegpu_test ID DESCRIPTION [extra flags...]
#
# Runs a 10-step RL training job on a single GPU using colocate_all + sleep.
# colocate_all packs all roles (actor + vLLM) into one placement group.
# enable_sleep lets vLLM release GPU memory while the actor trains, so they
# time-share without OOM.
#
# Base config differences from the multi-GPU suite:
#   - ray start --num-gpus=1 (instead of 2)
#   - --actor.num_gpus_per_node 1
#   - --vllm.num_engines 1 --vllm.tensor_parallel_size 1
#   - --train.colocate_all --vllm.enable_sleep --ds.enable_sleep
#   - --vllm.gpu_memory_utilization 0.4 (leave room for actor)
#
# PASS criteria: exit 0 AND at least 1 "Global step" line in the log.
run_rl_singlegpu_test() {
    local id=$1 desc=$2; shift 2
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"
        echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }

    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    ray_start
    "$VENV/bin/python" -m openrlhf.cli.train_ppo_ray \
        --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
        --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
        --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
        --train.colocate_all --vllm.enable_sleep --ds.enable_sleep \
        --vllm.sync_backend gloo \
        --actor.model_name_or_path "$MODEL" \
        --reward.remote_url "$REWARD_FN" \
        --data.prompt_dataset "$PROMPTS" \
        --data.input_key prompt --data.label_key label --data.apply_chat_template \
        --data.max_len 512 --data.max_samples 80 \
        --train.batch_size 8 --train.micro_batch_size 4 \
        --train.max_epochs 1 \
        --ds.attn_implementation sdpa \
        --ckpt.output_dir /tmp/sg_"$id" --ckpt.save_steps -1 \
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
# Runs SFT / RM / DPO directly via DeepSpeed — no Ray, no vLLM, 1 GPU.
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
log "OpenRLHF XPU E2E Test Suite — Single GPU  $TS"
log "Hardware: 1 GPU  | Model: $MODEL"
log "Results:  $RESULTS"
log "══════════════════════════════════════════════"

# ── Group 1: GRPO (colocate_all + sleep) ──────────────────────────────────────
# Actor and vLLM share the single GPU via Ray placement groups.
# vllm.enable_sleep and ds.enable_sleep allow them to time-share memory:
# vLLM releases memory during the training step, actor releases it during
# rollout generation. gpu_memory_utilization=0.4 leaves headroom for both.

run_rl_singlegpu_test "sg_grpo_baseline" \
    "GRPO, ZeRO-2, colocate_all + sleep — single-GPU baseline" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_grpo_lora" \
    "GRPO, LoRA rank=16, colocate_all + sleep — adapter-only training on 1 GPU" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload \
    --ds.lora.rank 16 --ds.lora.alpha 32

run_rl_singlegpu_test "sg_grpo_kl" \
    "GRPO + KL penalty (k3, use_loss), colocate_all + sleep" \
    --algo.advantage.estimator group_norm \
    --algo.kl.init_coef 0.01 --algo.kl.use_loss --algo.kl.estimator k3 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 2: Other RL algorithms ──────────────────────────────────────────────
# Same colocate_all + sleep pattern; only the advantage estimator changes.

run_rl_singlegpu_test "sg_reinforce" \
    "REINFORCE, colocate_all + sleep — simplest policy gradient on 1 GPU" \
    --algo.advantage.estimator reinforce --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 1 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_reinforce_baseline" \
    "REINFORCE++ baseline, colocate_all + sleep — with per-group mean baseline" \
    --algo.advantage.estimator reinforce_baseline --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_rloo" \
    "RLOO, colocate_all + sleep — leave-one-out baseline" \
    --algo.advantage.estimator rloo --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_dr_grpo" \
    "DR-GRPO, colocate_all + sleep — GRPO without std-deviation normalisation" \
    --algo.advantage.estimator dr_grpo --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 3: Reward shaping ───────────────────────────────────────────────────

run_rl_singlegpu_test "sg_grpo_overlong_penalty" \
    "GRPO + overlong penalty, colocate_all + sleep" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --reward.overlong_buffer_len 50 --reward.overlong_penalty_factor 1.0 \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_grpo_reward_norm" \
    "GRPO + reward normalisation, colocate_all + sleep" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --reward.normalize_enable \
    --ds.zero_stage 2 --ds.adam_offload

run_rl_singlegpu_test "sg_grpo_is_correction" \
    "GRPO + IS correction, colocate_all + sleep — token-level importance sampling" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --algo.advantage.is_correction_enable \
    --algo.advantage.is_correction_type tis \
    --algo.advantage.is_correction_threshold 0.5 5.0 \
    --ds.zero_stage 2 --ds.adam_offload

# ── Group 4: EMA checkpoint ───────────────────────────────────────────────────

run_rl_singlegpu_test "sg_grpo_ema" \
    "GRPO + EMA, colocate_all + sleep — exponential moving average on 1 GPU" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --train.enable_ema --train.ema_beta 0.992 \
    --ds.zero_stage 2 --ds.adam_offload

# NOTE: Async GRPO is intentionally NOT in the single-GPU suite. Async mode
# overlaps vLLM generation with actor training, which requires them on separate
# devices (>=2 GPUs): async + vLLM sleep mode is disallowed by OpenRLHF, and
# async + colocate_all without sleep deadlocks on a single device. It is
# validated as `grpo_async` in test_e2e_suite_multigpu.sh.

# ── Group 5: SFT ──────────────────────────────────────────────────────────────
# Pure DeepSpeed — no Ray, no vLLM. Already 1-GPU by design.

run_supervised_test "sg_sft_zero2" \
    "SFT, ZeRO-2, full fine-tune on 1 GPU" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/sg_sft_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sg_sft_lora" \
    "SFT, LoRA rank=16 on 1 GPU — adapter-only, base model frozen" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/sg_sft_lora --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sg_sft_packing" \
    "SFT + sample packing on 1 GPU — eliminates padding waste" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.packing_samples \
    --ckpt.output_dir /tmp/sg_sft_packing --ckpt.save_steps -1 \
    --eval.steps -1

# ── Group 6: Reward Model + DPO / IPO / cDPO ─────────────────────────────────

run_supervised_test "sg_rm_zero2" \
    "Reward Model, ZeRO-2 on 1 GPU — preference pair binary cross-entropy" \
    train_rm \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/sg_rm_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sg_dpo_zero2" \
    "DPO, ZeRO-2 on 1 GPU — direct preference optimisation" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ckpt.output_dir /tmp/sg_dpo_zero2 --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sg_dpo_ipo" \
    "IPO on 1 GPU — avoids DPO overoptimisation" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --model.ipo_enable \
    --ckpt.output_dir /tmp/sg_dpo_ipo --ckpt.save_steps -1 \
    --eval.steps -1

run_supervised_test "sg_dpo_cdpo" \
    "cDPO on 1 GPU — label smoothing for noisy preference labels" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --model.label_smoothing 0.1 \
    --ckpt.output_dir /tmp/sg_dpo_cdpo --ckpt.save_steps -1 \
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
