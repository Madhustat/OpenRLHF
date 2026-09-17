#!/usr/bin/env bash
# =============================================================================
# OpenRLHF XPU E2E Test Suite — Single GPU, EXTENDED (gap coverage)
# =============================================================================
#
# This is a SEPARATE suite that extends `test_e2e_suite_singlegpu.sh`. It reuses
# that suite's exact helpers, base config and PASS criteria — nothing is
# redefined — and adds the 26 cases that suite does not cover.
#
# Run the base suite first (18 cases), then this one (26 cases) = 44 total.
#
#   bash tests/test_e2e_suite_singlegpu.sh            # cases 1-18  (already PASS)
#   bash tests/test_e2e_suite_singlegpu_extended.sh   # cases 19-44 (new)
#   bash tests/test_e2e_suite_singlegpu_extended.sh sg_ppo_gae   # single case
#
# -----------------------------------------------------------------------------
# ZeRO STAGE: every case runs at stage 2, same as the base suite.
#
# Rationale (measured, not assumed): at world_size=1 ZeRO's partitioning is a
# no-op — the same model occupies 7226 MiB at ws=1 versus 4090 MiB/rank at ws=2.
# The ONLY reason stage >= 1 matters on one device is that `--ds.enable_sleep`
# requires it: stage 0 builds FP16_UnfusedOptimizer, which has no
# offload_states() method at all. So the stage number is not a test axis here
# and is not encoded in any case name.
#
# -----------------------------------------------------------------------------
# CASES PROPOSED AND REJECTED — validated as impossible, recorded so they are
# not re-proposed:
#
#   critic CPU offload   `--critic.offload` DOES NOT EXIST. Only `--ref.offload`
#                        and `--reward.offload` are defined in train_ppo_ray.py.
#
#   partial rollout      `--train.partial_rollout_enable` asserts
#                        `--train.async_enable` (train_ppo_ray.py:683-684).
#                        Async asserts `not --vllm.enable_sleep` and is rejected
#                        with colocate_all. colocate_all is mandatory on one
#                        device, so this is structurally unreachable.
#
#   vLLM over-sampling   `--rollout.vllm_generate_batch_size >
#                        --rollout.batch_size` asserts async too (:705-709).
#
#   ZeRO-3 cases         Dropped deliberately. At ws=1 stage 3 gains nothing over
#                        stage 2 except `offload_param` (unverified at runtime),
#                        and the two ZeRO-3 fixes (overlap_comm,
#                        FusedAdam-capability) are not carried on the single-GPU
#                        branch. Do not add stage-3 cases here without also
#                        porting those fixes.
#
#   anything multi-device  --train.async_enable, --ds.tensor_parallel_size>1,
#                        --ds.ring_attn_size>1, --vllm.tensor_parallel_size>1,
#                        --vllm.num_engines>1, --actor.num_gpus_per_node>1.
#                        All need >=2 physical devices.
# =============================================================================
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
RAY=${RAY:-ray}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
VLM_MODEL=${VLM_MODEL:-Qwen/Qwen2-VL-2B-Instruct}
REWARD_FN=$REPO/examples/python/math_reward_func.py
AGENT_FN=$REPO/examples/python/agent_func.py
PROMPTS=${PROMPTS:-$REPO/tests/data/gsm8k_train_prompts.jsonl}
SFT_DATA=${SFT_DATA:-$REPO/tests/data/gsm8k_sft/train.parquet}
PREF_DATA=OpenRLHF/preference_dataset_mixture2_and_safe_pku
VLM_DATA=${VLM_DATA:-hiyouga/geometry3k}

FILTER=${1:-}
RESULTS=$REPO/tests/results/singlegpu_extended_$(date +%Y%m%d_%H%M%S)
SUMMARY=$RESULTS/summary.txt
mkdir -p "$RESULTS"
PASS=0; FAIL=0; SKIP=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS/run.log"; }

# Auto-generate datasets if missing (same as the base suite).
if [[ ! -f "$PROMPTS" || ! -f "$SFT_DATA" ]]; then
    log "Generating E2E datasets..."
    "$PYTHON" "$REPO/tests/prepare_e2e_data.py" || exit 1
fi

_cleanup_procs() {
    "$RAY" stop --force >/dev/null 2>&1 || true
    pkill -9 -f "train_ppo_ray|train_sft|train_dpo|train_rm|EngineCore|PolicyModelActor|RolloutRayActor|ray::" \
        >/dev/null 2>&1 || true
    rm -rf /tmp/ray
}
_xpu_health() {
    # Log XPU health before each case so a wedged device is attributable to the prior case.
    if ONEAPI_DEVICE_SELECTOR=level_zero:0 timeout 30 "$PYTHON" -c \
        "import torch;t=torch.ones(2,device='xpu');torch.xpu.synchronize();assert float(t.sum())==2.0" \
        >/dev/null 2>&1; then
        log "[HEALTH] XPU OK"
    else
        log "[HEALTH] WARNING: XPU not responsive (device may be wedged)"
    fi
}
ray_start() {
    _cleanup_procs
    sleep 2
    _xpu_health
    "$RAY" start --head --num-gpus=1 --disable-usage-stats >/dev/null 2>&1
    sleep 4
}
ray_stop() { _cleanup_procs; sleep 4; }

# -----------------------------------------------------------------------------
# run_rl_singlegpu_test ID DESCRIPTION [extra flags...]
#
# IDENTICAL base config to test_e2e_suite_singlegpu.sh's helper of the same
# name, so results are comparable across the two suites.
# PASS criteria: exit 0 AND at least 1 "Global step" line in the log.
# -----------------------------------------------------------------------------
run_rl_singlegpu_test() {
    local id=$1 desc=$2; shift 2
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"; echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }
    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    ray_start
    "$PYTHON" -m openrlhf.cli.train_ppo_ray \
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

    # Resume-aware criterion: a resume run started from the FINAL saved step has no
    # prompts left, so it legitimately logs 0 new "Global step" lines. Counting steps
    # would false-FAIL a working resume. Instead verify the checkpoint was actually
    # loaded and a non-zero global_step was restored.
    if [[ "$id" == *ckpt_resume* ]]; then
        local loaded gstep
        loaded=$(grep -c "Loading the checkpoint" "$logf" 2>/dev/null || echo 0)
        gstep=$(grep -oE "'global_step': [0-9]+" "$logf" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
        gstep=${gstep:-0}
        if [[ $rc -eq 0 && $loaded -ge 1 && $gstep -ge 1 ]]; then
            log "PASS  $id (resumed from global_step=$gstep, +$steps new steps)"
            echo "PASS  $id — $desc (resumed from global_step=$gstep, +$steps new steps)" >> "$SUMMARY"; ((PASS++))
        else
            log "FAIL  $id (exit=$rc resume_loaded=$loaded global_step=$gstep)"
            echo "FAIL  $id — $desc (exit=$rc, resume_loaded=$loaded, global_step=$gstep)" >> "$SUMMARY"; ((FAIL++))
        fi
        return
    fi

    # Load-and-evaluate-only: 0 training steps by design (data exhausted at saved
    # step); PASS on checkpoint loaded AND eval actually ran on the loaded weights.
    if [[ "$id" == *ckpt_eval_only* ]]; then
        local loaded evaled
        loaded=$(grep -c "Loading the checkpoint" "$logf" 2>/dev/null || echo 0)
        evaled=$(grep -cE "Evaluation completed|Eval-on-load" "$logf" 2>/dev/null || echo 0)
        if [[ $rc -eq 0 && $loaded -ge 1 && $evaled -ge 1 ]]; then
            log "PASS  $id (loaded ckpt + eval-on-load ran)"
            echo "PASS  $id — $desc (loaded ckpt + eval-on-load ran)" >> "$SUMMARY"; ((PASS++))
        else
            log "FAIL  $id (exit=$rc loaded=$loaded eval_ran=$evaled)"
            echo "FAIL  $id — $desc (exit=$rc, loaded=$loaded, eval_ran=$evaled)" >> "$SUMMARY"; ((FAIL++))
        fi
        return
    fi

    if [[ $rc -eq 0 && $steps -ge 1 ]]; then
        log "PASS  $id ($steps steps)"
        echo "PASS  $id — $desc ($steps steps)" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id (exit=$rc steps=$steps)"
        echo "FAIL  $id — $desc (exit=$rc, steps=$steps)" >> "$SUMMARY"; ((FAIL++))
    fi
}

# -----------------------------------------------------------------------------
# run_rl_nobase_test ID DESCRIPTION [full flag list...]
#
# For cases that must OVERRIDE a base-config flag (e.g. drop --ds.enable_sleep,
# or drop --reward.remote_url). Bash cannot un-set a flag already emitted, so
# these cases spell out the whole invocation. Base config is otherwise copied
# verbatim — keep the two in sync if the base suite changes.
# -----------------------------------------------------------------------------
run_rl_nobase_test() {
    local id=$1 desc=$2; shift 2
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"; echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }
    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    ray_start
    "$PYTHON" -m openrlhf.cli.train_ppo_ray "$@" > "$logf" 2>&1
    local rc=$?
    ray_stop

    local steps
    steps=$(grep -c "Global step" "$logf" 2>/dev/null || echo 0)

    # Resume-aware criterion: a resume run started from the FINAL saved step has no
    # prompts left, so it legitimately logs 0 new "Global step" lines. Counting steps
    # would false-FAIL a working resume. Instead verify the checkpoint was actually
    # loaded and a non-zero global_step was restored.
    if [[ "$id" == *ckpt_resume* ]]; then
        local loaded gstep
        loaded=$(grep -c "Loading the checkpoint" "$logf" 2>/dev/null || echo 0)
        gstep=$(grep -oE "'global_step': [0-9]+" "$logf" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
        gstep=${gstep:-0}
        if [[ $rc -eq 0 && $loaded -ge 1 && $gstep -ge 1 ]]; then
            log "PASS  $id (resumed from global_step=$gstep, +$steps new steps)"
            echo "PASS  $id — $desc (resumed from global_step=$gstep, +$steps new steps)" >> "$SUMMARY"; ((PASS++))
        else
            log "FAIL  $id (exit=$rc resume_loaded=$loaded global_step=$gstep)"
            echo "FAIL  $id — $desc (exit=$rc, resume_loaded=$loaded, global_step=$gstep)" >> "$SUMMARY"; ((FAIL++))
        fi
        return
    fi

    # Load-and-evaluate-only: 0 training steps by design (data exhausted at saved
    # step); PASS on checkpoint loaded AND eval actually ran on the loaded weights.
    if [[ "$id" == *ckpt_eval_only* ]]; then
        local loaded evaled
        loaded=$(grep -c "Loading the checkpoint" "$logf" 2>/dev/null || echo 0)
        evaled=$(grep -cE "Evaluation completed|Eval-on-load" "$logf" 2>/dev/null || echo 0)
        if [[ $rc -eq 0 && $loaded -ge 1 && $evaled -ge 1 ]]; then
            log "PASS  $id (loaded ckpt + eval-on-load ran)"
            echo "PASS  $id — $desc (loaded ckpt + eval-on-load ran)" >> "$SUMMARY"; ((PASS++))
        else
            log "FAIL  $id (exit=$rc loaded=$loaded eval_ran=$evaled)"
            echo "FAIL  $id — $desc (exit=$rc, loaded=$loaded, eval_ran=$evaled)" >> "$SUMMARY"; ((FAIL++))
        fi
        return
    fi

    if [[ $rc -eq 0 && $steps -ge 1 ]]; then
        log "PASS  $id ($steps steps)"
        echo "PASS  $id — $desc ($steps steps)" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id (exit=$rc steps=$steps)"
        echo "FAIL  $id — $desc (exit=$rc, steps=$steps)" >> "$SUMMARY"; ((FAIL++))
    fi
}

# Flags common to run_rl_nobase_test cases, MINUS the sleep/reward flags each
# case decides for itself.
rl_base_min() {
    echo --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
        --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
        --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
        --train.colocate_all --vllm.sync_backend gloo \
        --actor.model_name_or_path "$MODEL" \
        --data.prompt_dataset "$PROMPTS" \
        --data.input_key prompt --data.label_key label --data.apply_chat_template \
        --data.max_len 512 --data.max_samples 80 \
        --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
        --ds.attn_implementation sdpa --logger.logging_steps 1 --eval.steps -1
}

# -----------------------------------------------------------------------------
# run_supervised_test ID DESCRIPTION TRAINER [extra flags...]
# Identical to the base suite's helper. PASS: exit 0 AND >=1 "loss=" line.
# -----------------------------------------------------------------------------
run_supervised_test() {
    local id=$1 desc=$2 trainer=$3; shift 3
    [[ -n "$FILTER" && "$id" != *"$FILTER"* ]] && {
        log "SKIP $id — filtered"; echo "SKIP  $id — filtered" >> "$SUMMARY"
        ((SKIP++)); return
    }
    local logf=$RESULTS/$id.log
    log "START $id — $desc"

    PYTHONUNBUFFERED=1 "$PYTHON" -m openrlhf.cli."$trainer" \
        --ds.zero_stage 2 --ds.adam_offload \
        --ds.attn_implementation sdpa \
        --ds.param_dtype bf16 \
        --train.max_epochs 1 \
        --logger.logging_steps 1 \
        "$@" > "$logf" 2>&1
    local rc=$?

    local losses
    losses=$(grep -c "loss=" "$logf" 2>/dev/null || echo 0)
    if [[ $rc -eq 0 && $losses -ge 1 ]]; then
        log "PASS  $id ($losses loss lines)"
        echo "PASS  $id — $desc ($losses loss lines)" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id (exit=$rc losses=$losses)"
        echo "FAIL  $id — $desc (exit=$rc, losses=$losses)" >> "$SUMMARY"; ((FAIL++))
    fi
}

GRPO_BASE=(--algo.advantage.estimator group_norm --algo.kl.init_coef 0
           --rollout.n_samples_per_prompt 4 --rollout.batch_size 8
           --ds.zero_stage 2 --ds.adam_offload)

log "══════════════════════════════════════════════"
log "Single-GPU EXTENDED suite — 26 cases (19-44)"
log "Hardware: 1 GPU  | Model: $MODEL"
log "Results:  $RESULTS"
log "══════════════════════════════════════════════"

# ═════════════════════════════════════════════════════════════════════════════
# A. CORE RL ALGORITHM COVERAGE
#    The base suite covers 5 of 6 advantage estimators. `gae` is the only one
#    never exercised — and the only one that instantiates a critic. This is the
#    single largest gap in single-GPU coverage.
# ═════════════════════════════════════════════════════════════════════════════

# [19] MUST-RUN. Adds a second trainable model (critic) on the one device.
#      If this OOMs, that is a memory-tuning result (lower
#      gpu_memory_utilization), not a failed feature.
run_rl_singlegpu_test "sg_ppo_gae" \
    "PPO with critic (GAE) — 6th/last advantage estimator, critic + value head" \
    --algo.advantage.estimator gae --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 1 --rollout.batch_size 8 \
    --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
    --ds.zero_stage 2   # adam_offload dropped: host-RAM-bound box (actor+critic roles), optimizer to VRAM

# [20]
run_rl_singlegpu_test "sg_ppo_gae_critic_freezing" \
    "PPO+critic, critic frozen for first 3 steps while actor trains" \
    --algo.advantage.estimator gae --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 1 --rollout.batch_size 8 \
    --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
    --critic.freezing_steps 3 \
    --ds.zero_stage 2   # adam_offload dropped: host-RAM-bound box (actor+critic roles), optimizer to VRAM

# ═════════════════════════════════════════════════════════════════════════════
# B. REWARD & REFERENCE PIPELINE
#    The base suite always uses a remote reward FUNCTION. A reward MODEL is a
#    different code path and loads an additional model onto the device.
# ═════════════════════════════════════════════════════════════════════════════

# [21] MUST-RUN. Drops --reward.remote_url, so uses run_rl_nobase_test.
run_rl_nobase_test "sg_grpo_reward_model" \
    "Reward MODEL instead of remote reward function — 4th model on device" \
    $(rl_base_min) --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_reward_model --ckpt.save_steps -1 \
    --reward.model_name_or_path "$MODEL" \
    --reward.num_nodes 1 --reward.num_gpus_per_node 1 \
    "${GRPO_BASE[@]}"

# [22]
run_rl_nobase_test "sg_grpo_reward_offload" \
    "Reward-model CPU offload (--reward.offload)" \
    $(rl_base_min) --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_reward_offload --ckpt.save_steps -1 \
    --reward.model_name_or_path "$MODEL" \
    --reward.num_nodes 1 --reward.num_gpus_per_node 1 --reward.offload \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2   # GRPO_BASE minus adam_offload: host-RAM-bound box (reward on host), optimizer to VRAM

# [23] KL coef > 0 so the reference model is actually used.
run_rl_singlegpu_test "sg_grpo_ref_offload" \
    "Reference-model CPU offload (--ref.offload) — frozen KL reference to host" \
    --algo.advantage.estimator group_norm \
    --algo.kl.init_coef 0.01 --algo.kl.estimator k3 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 --ref.offload \
    --ds.zero_stage 2 --ds.adam_offload

# ═════════════════════════════════════════════════════════════════════════════
# C. MEMORY MANAGEMENT — the defining single-GPU constraint
#    The base suite runs exactly one point in this space: both sleeps ON and
#    adam_offload ON. Everything below is unexplored.
# ═════════════════════════════════════════════════════════════════════════════

# [24] Establishes the no-sleep memory ceiling. A clean OOM is a valid result.
run_rl_nobase_test "sg_grpo_no_sleep" \
    "colocate_all with BOTH sleeps OFF — no-sleep memory ceiling" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --ckpt.output_dir /tmp/sg_grpo_no_sleep --ckpt.save_steps -1 \
    "${GRPO_BASE[@]}"

# [25]
run_rl_nobase_test "sg_grpo_vllm_sleep_only" \
    "vLLM sleep ON, DeepSpeed sleep OFF" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" --vllm.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_vllm_sleep_only --ckpt.save_steps -1 \
    "${GRPO_BASE[@]}"

# [26]
run_rl_nobase_test "sg_grpo_ds_sleep_only" \
    "DeepSpeed sleep ON, vLLM sleep OFF" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_ds_sleep_only --ckpt.save_steps -1 \
    "${GRPO_BASE[@]}"

# [27] MUST-RUN, and the most important case in this file.
#      offload_deepspeed_states() returns early when adam_offload is on, so
#      --ds.enable_sleep is a SILENT NO-OP in every one of the base suite's
#      sleep tests. This is the only configuration where DeepSpeed sleep
#      actually offloads optimizer state.
run_rl_nobase_test "sg_grpo_no_adam_offload" \
    "adam_offload OFF — the ONLY config where DS-sleep actually offloads" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_no_adam_offload --ckpt.save_steps -1 \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
    --ds.zero_stage 2

# [28] Worst case: 4 models resident, nothing sleeping. Expect memory pressure.
run_rl_nobase_test "sg_ppo_gae_no_sleep" \
    "PPO+critic with both sleeps OFF — worst-case residency, 4 models" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --ckpt.output_dir /tmp/sg_ppo_gae_no_sleep --ckpt.save_steps -1 \
    --algo.advantage.estimator gae --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 1 --rollout.batch_size 8 \
    --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
    --ds.zero_stage 2 --ds.adam_offload

# ═════════════════════════════════════════════════════════════════════════════
# D. THROUGHPUT & BATCHING
# ═════════════════════════════════════════════════════════════════════════════

# [29] MUST-RUN. The base suite covers SFT packing only; the RL replay-buffer
#      packing path is separate code.
# adam_offload dropped: host-RAM-bound single XPU (31GB) -> keep optimizer on idle VRAM.
# (Requires transformers>=5.16 so packing doesn't hit the flash-attn2/torch-2.13 kernel gap.)
run_rl_singlegpu_test "sg_grpo_packing" \
    "Sample packing in the RL path (--ds.packing_samples)" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 --ds.zero_stage 2 \
    --ds.packing_samples

# [30]
# Reduced-memory settings for the 31GB single-XPU host-RAM ceiling: halve the dynamic
# token budget (16192->8192), n_samples (4->2), max_len (512->384), fewer async tasks,
# fewer samples. Full-size dynamic batching needs more host RAM / >1 GPU.
export OPENRLHF_ASYNC_NUM_TASKS=4
run_rl_singlegpu_test "sg_grpo_dynamic_batch" \
    "Dynamic token-budgeted batching (--train.dynamic_batch_enable) — reduced mem for single XPU" \
    --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
    --rollout.n_samples_per_prompt 2 --rollout.batch_size 8 \
    --ds.zero_stage 2 --ds.adam_offload \
    --train.dynamic_batch_enable --train.max_tokens_per_gpu 8192 \
    --data.max_len 384 --data.max_samples 40
unset OPENRLHF_ASYNC_NUM_TASKS

# [31]
run_rl_singlegpu_test "sg_grpo_grad_ckpt" \
    "Gradient checkpointing (--actor.gradient_checkpointing_enable)" \
    "${GRPO_BASE[@]}" --actor.gradient_checkpointing_enable

# ═════════════════════════════════════════════════════════════════════════════
# E. ROLLOUT TUNING
# ═════════════════════════════════════════════════════════════════════════════

# [32] Relevant because n_samples_per_prompt=4 shares prompt prefixes.
run_rl_singlegpu_test "sg_grpo_prefix_caching" \
    "vLLM prefix caching (--vllm.enable_prefix_caching)" \
    "${GRPO_BASE[@]}" --vllm.enable_prefix_caching

# [33] Asserts (train_ppo_ray.py:710-720): range[0] < range[1], remote_url or
#      agent_func_path set, n_samples_per_prompt > 1. All satisfied here.
run_rl_singlegpu_test "sg_grpo_dynamic_filtering" \
    "Dynamic prompt filtering by reward range (--algo.dynamic_filtering_enable)" \
    "${GRPO_BASE[@]}" --algo.dynamic_filtering_enable \
    --algo.dynamic_filtering_range 0 1

# Upstream examples/scripts parity — single-XPU versions of the two advanced recipes.
# Both use a KL reference model: --ref.num_nodes/num_gpus_per_node 1 required under
# colocation (else ref defaults to 8 GPUs -> assert). adam_offload dropped (host-RAM).
# Dynamic filtering discards no-reward-variance batches, so step count may be < 10 -- expected.
run_rl_singlegpu_test "sg_dapo" \
    "DAPO (single XPU) — GRPO + dynamic filtering + clip-higher + KL-loss" \
    --algo.advantage.estimator group_norm --algo.advantage.gamma 1.0 \
    --actor.eps_clip_low_high 0.2 0.28 \
    --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0 1 \
    --algo.kl.init_coef 0.001 --algo.kl.use_loss --algo.kl.estimator k3 \
    --rollout.n_samples_per_prompt 8 --rollout.batch_size 8 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
    --ds.zero_stage 2 --data.max_samples 160

run_rl_singlegpu_test "sg_prorlv2" \
    "ProRL-v2 (single XPU) — REINFORCE++ + dynamic filtering + clip-higher + KL-loss" \
    --algo.advantage.estimator reinforce_baseline --algo.advantage.gamma 1.0 \
    --actor.eps_clip_low_high 0.2 0.28 \
    --algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0 1 \
    --algo.kl.init_coef 0.0001 --algo.kl.use_loss --algo.kl.estimator k2 \
    --rollout.n_samples_per_prompt 8 --rollout.batch_size 8 \
    --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
    --ds.zero_stage 2 --data.max_samples 160

# ═════════════════════════════════════════════════════════════════════════════
# F. TRAINING LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════════

# [34] Two invocations: save, then resume. Distinct code path from a fresh run.
#      VERIFY MANUALLY: the resumed run must start from the saved step, not 0.
run_rl_nobase_test "sg_grpo_ckpt_save" \
    "Checkpoint SAVE at step 5 (part 1 of 2)" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_ckpt --ckpt.save_steps 5 \
    "${GRPO_BASE[@]}"

run_rl_nobase_test "sg_grpo_ckpt_resume" \
    "Checkpoint RESUME from step 5 (part 2 of 2)" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_ckpt --ckpt.save_steps 5 --ckpt.load_enable \
    "${GRPO_BASE[@]}"

# [34b] Load checkpoint, CONTINUE training, and run in-training eval — exercises the
#       loaded weights across post-load steps (larger dataset so steps actually run).
run_rl_nobase_test "sg_grpo_ckpt_load_eval" \
    "Load checkpoint, continue training + in-training eval (loaded weights used across steps)" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_ckpt --ckpt.save_steps 5 --ckpt.load_enable \
    --data.max_samples 160 \
    --eval.steps 5 --eval.dataset "$PROMPTS" \
    "${GRPO_BASE[@]}"

# [34c] PURE "load saved model and evaluate": eval the LOADED weights before any
#       training (OPENRLHF_EVAL_ON_LOAD=1 hook). Dataset is exhausted at the saved
#       step, so 0 training steps — the eval reflects the loaded weights alone.
export OPENRLHF_EVAL_ON_LOAD=1
run_rl_nobase_test "sg_grpo_ckpt_eval_only" \
    "Load saved checkpoint then EVALUATE it only (eval-on-load, no training)" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_ckpt --ckpt.save_steps 5 --ckpt.load_enable \
    --eval.steps -1 --eval.dataset "$PROMPTS" \
    "${GRPO_BASE[@]}"
unset OPENRLHF_EVAL_ON_LOAD

# [35] The artifact anything downstream actually consumes.
run_rl_nobase_test "sg_grpo_save_hf" \
    "HF-format model export (--ckpt.save_hf)" \
    $(rl_base_min) --reward.remote_url "$REWARD_FN" \
    --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_save_hf --ckpt.save_steps 5 --ckpt.save_hf \
    "${GRPO_BASE[@]}"

# [36] Assert (train_ppo_ray.py:687): --eval.dataset requires remote_url or
#      agent_func_path. remote_url is set, so satisfied.
run_rl_nobase_test "sg_grpo_eval" \
    "In-training eval path (--eval.steps + --eval.dataset)" \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
    --train.colocate_all --vllm.enable_sleep --ds.enable_sleep \
    --vllm.sync_backend gloo \
    --actor.model_name_or_path "$MODEL" --reward.remote_url "$REWARD_FN" \
    --data.prompt_dataset "$PROMPTS" \
    --data.input_key prompt --data.label_key label --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 80 \
    --train.batch_size 8 --train.micro_batch_size 4 --train.max_epochs 1 \
    --ds.attn_implementation sdpa --logger.logging_steps 1 \
    --ckpt.output_dir /tmp/sg_grpo_eval --ckpt.save_steps -1 \
    --eval.steps 5 --eval.dataset "$PROMPTS" \
    "${GRPO_BASE[@]}"

# ═════════════════════════════════════════════════════════════════════════════
# G. SUPERVISED WORKFLOW VARIANTS
#    The base suite covers SFT full/LoRA/packing, RM full, DPO full/IPO/cDPO.
#    Missing: RM LoRA, RM packing, DPO LoRA, DPO packing.
# ═════════════════════════════════════════════════════════════════════════════

# [37]
run_supervised_test "sg_rm_lora" \
    "Reward Model + LoRA rank=16 on 1 GPU" \
    train_rm \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/sg_rm_lora --ckpt.save_steps -1 \
    --eval.steps -1

# [38]
run_supervised_test "sg_rm_packing" \
    "Reward Model + sample packing on 1 GPU" \
    train_rm \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.packing_samples \
    --ckpt.output_dir /tmp/sg_rm_packing --ckpt.save_steps -1 \
    --eval.steps -1

# [39]
run_supervised_test "sg_dpo_lora" \
    "DPO + LoRA rank=16 on 1 GPU" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/sg_dpo_lora --ckpt.save_steps -1 \
    --eval.steps -1

# [40]
run_supervised_test "sg_dpo_packing" \
    "DPO + sample packing on 1 GPU" \
    train_dpo \
    --model.model_name_or_path "$MODEL" \
    --ref.model_name_or_path "$MODEL" \
    --data.dataset "$PREF_DATA" \
    --data.chosen_key chosen --data.rejected_key rejected \
    --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.packing_samples \
    --ckpt.output_dir /tmp/sg_dpo_packing --ckpt.save_steps -1 \
    --eval.steps -1

# ═════════════════════════════════════════════════════════════════════════════
# H. SCOPE DISCOVERY — never run on XPU before.
#    These are EXPECTED to possibly fail. The goal is to learn HOW they fail,
#    not to debug them. Capture the error and move on.
# ═════════════════════════════════════════════════════════════════════════════

# [41] Note: the only shipped example for this is async
#      (examples/scripts/train_reinforce_baseline_ray_agent_async.sh). No assert
#      requires async, but the sync path may be less exercised upstream.
run_rl_nobase_test "sg_grpo_agent_func" \
    "SCOPE DISCOVERY: agent-based multi-turn rollout (--train.agent_func_path)" \
    $(rl_base_min) --vllm.enable_sleep --ds.enable_sleep \
    --ckpt.output_dir /tmp/sg_grpo_agent_func --ckpt.save_steps -1 \
    --train.agent_func_path "$AGENT_FN" \
    "${GRPO_BASE[@]}"

# [42] Hard asserts: VLM does NOT support a critic (so gae is impossible) and
#      does NOT support --ds.packing_samples. 2B model on one device.
run_rl_nobase_test "sg_grpo_vlm" \
    "SCOPE DISCOVERY: vision-language RL (no critic, no packing — asserted)" \
    --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
    --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
    --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
    --train.colocate_all --vllm.enable_sleep --ds.enable_sleep \
    --vllm.sync_backend gloo \
    --actor.model_name_or_path "$VLM_MODEL" --reward.remote_url "$REWARD_FN" \
    --data.prompt_dataset "$VLM_DATA" \
    --data.input_key problem --data.label_key answer --data.apply_chat_template \
    --data.image_key images --data.max_images_per_prompt 1 \
    `# VLM: image tokens expand prompts to ~900-1200 -> 512 truncates & breaks pixel alignment` \
    --data.max_len 2048 --data.max_samples 40 \
    --train.batch_size 8 --train.micro_batch_size 1 --train.max_epochs 1 \
    --ds.attn_implementation sdpa --logger.logging_steps 1 --eval.steps -1 \
    --ckpt.output_dir /tmp/sg_grpo_vlm --ckpt.save_steps -1 \
    "${GRPO_BASE[@]}"

# [43] Uses DeepSpeed's MuonWithAuxAdam. NOT FusedAdam — so with ZeRO-3 +
#      DS-sleep it would hit the FusedAdam-only offload_states assert.
#      Irrelevant at stage 2, but relevant if the stage is ever raised.
run_rl_singlegpu_test "sg_grpo_muon" \
    "SCOPE DISCOVERY: Muon optimizer (--actor.optim muon) instead of AdamW" \
    "${GRPO_BASE[@]}" --actor.optim muon

# [44] 4-bit needs bitsandbytes, which is CUDA-oriented. Expected to fail on
#      XPU; the point is to capture HOW, and to confirm it does not silently
#      pull in a CUDA torch.
run_supervised_test "sg_sft_qlora" \
    "SCOPE DISCOVERY: QLoRA / 4-bit (--ds.load_in_4bit + LoRA) on SFT" \
    train_sft \
    --model.model_name_or_path "$MODEL" \
    --data.dataset "$SFT_DATA" \
    --data.input_key messages --data.apply_chat_template \
    --data.max_len 512 --data.max_samples 128 \
    --train.batch_size 8 --train.micro_batch_size 2 \
    --ds.load_in_4bit --ds.lora.rank 16 --ds.lora.alpha 32 \
    --ckpt.output_dir /tmp/sg_sft_qlora --ckpt.save_steps -1 \
    --eval.steps -1

# ═════════════════════════════════════════════════════════════════════════════
log "══════════════════════════════════════════════"
log "EXTENDED suite done: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
log "Summary: $SUMMARY"
log "══════════════════════════════════════════════"
cat "$SUMMARY"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
