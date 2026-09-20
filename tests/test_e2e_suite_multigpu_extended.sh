#!/usr/bin/env bash
# =============================================================================
# OpenRLHF XPU E2E Test Suite — TWO GPUs, EXTENDED (gap coverage)
# =============================================================================
#
# A SEPARATE suite extending `test_e2e_suite_multigpu.sh`. It re-declares that
# suite's config and PASS criteria verbatim (not sourced — sourcing would run its
# 28 cases) and adds the behaviours no 2-GPU suite covers today.
#
# Coverage was established objectively, not by assertion: MASTER_COVERAGE.py
# enumerates 133 testable behaviours across all 6 entrypoints and resolves each
# one's status from recorded results. 74 already PASS (upstream bench + 80-cell
# gloo matrix), 8 exist in the base e2e suite but have never run on this tree,
# and the 55 cases below close the remaining 51 behaviours.
#
#   bash tests/test_e2e_suite_multigpu.sh              # base 28 cases
#   bash tests/test_e2e_suite_multigpu_extended.sh     # these 55
#   bash tests/test_e2e_suite_multigpu_extended.sh mg_gspo    # one case
#
# -----------------------------------------------------------------------------
# WHAT IS DELIBERATELY NOT HERE, and why
#
#   topology / sleep / ZeRO stage    The 80-cell gloo matrix sweeps all of it:
#                                    2 engines x TP1, 1 engine x TP2, critic
#                                    ws=1 and ws=2, 4 sleep modes, stages 0-3.
#                                    63/63 passable cells PASS. Re-testing here
#                                    would add hours and no information.
#
#   all 7 advantage estimators       gae, reinforce, reinforce_baseline, rloo,
#                                    group_norm, dr_grpo, flash_reinforce are
#                                    already covered by the bench + base suite.
#
#   packing, dynamic batching,       covered by the base e2e suite and/or the
#   LoRA rank, EMA, IS correction,   upstream-scripts bench.
#   KL-as-loss, overlong penalty,
#   reward norm, agent, VLM,
#   async, partial rollout
#
# -----------------------------------------------------------------------------
# EXPECTED NON-PASSES. These are scope discovery, not regressions. Treat them
# like the matrix's "Unsupported" cells: a FAIL here is a finding, not a defect.
#
#   mg_ds_autotp2      DeepSpeed AutoTP on XPU is unproven
#   mg_ring_attn2      ring attention needs a flash-attn kernel
#   mg_liger           Liger kernels are CUDA-oriented
#   mg_deepcompile     DeepCompile is unproven on XPU
#   mg_sync_with_ray   alternative weight-sync path, never exercised on XPU
#   mg_flash_attn2     kernels-community publishes no XPU flash-attn2 for torch 2.13
#   mg_moe_experts_grouped_mm   needs a grouped-GEMM kernel for this backend
#   mg_moe_experts_deepgemm     DeepGEMM is a CUDA-only kernel library
#
# -----------------------------------------------------------------------------
# MEASURED FACTS THAT SHAPE SPECIFIC CASES
#
#   gamma is IGNORED unless estimator=reinforce. experience_maker.py:308 resets
#   it to 1.0 for every other estimator and logs a warning. So mg_gamma uses
#   reinforce, otherwise the case would silently test nothing.
#
#   lambd only matters for gae, which requires a critic -> mg_lambda_gae adds one.
#
#   aux_loss_coef and experts_implementation are NO-OPS on a dense model, so the
#   MoE cases use granite-3.1-1b-a400m (32 experts, top-8) instead of Qwen2.5-0.5B.
#   Testing them on a dense model would produce a green cell that proves nothing.
#
#   A reward MODEL cannot be appended to the remote-reward-function base config;
#   --reward.remote_url and --reward.model_name_or_path are alternatives. Hence
#   the separate run_rl_rm helper rather than extra flags on run_rl.
# =============================================================================
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
RAY=${RAY:-ray}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
# A REAL Mixture-of-Experts model, not the dense default. Needed because
# --actor.aux_loss_coef and --ds.experts_implementation are no-ops on a dense
# model: there is no router to balance and no expert kernel to select, so a
# "PASS" on Qwen2.5-0.5B would prove nothing. VERIFIED on this box: loads on
# XPU, 1.33B params / 2.7 GB bf16, 32 experts top-8, routers present, forward OK.
# vLLM 0.27 registers GraniteMoeForCausalLM, so the rollout side works too.
# -instruct, not -base: the base checkpoint ships no tokenizer.chat_template, and
# the RL base config uses --data.apply_chat_template. MEASURED: -base fails with
# "Cannot use chat template functions because tokenizer.chat_template is not set".
MOE_MODEL=${MOE_MODEL:-ibm-granite/granite-3.1-1b-a400m-instruct}
# A SECOND, tiny MoE (2 layers, hidden 64, 4 experts) for the two cases granite
# cannot serve. Random weights, so the losses are meaningless -- but these cases
# test that a CODE PATH executes, not that training converges.
#   1. batched_mm OOMs on granite by construction (see mg_moe_experts_batched_mm)
#   2. transformers' GraniteMoE does not populate aux_loss (see mg_moe_aux_loss)
MOE_SMALL=${MOE_SMALL:-hf-internal-testing/tiny-random-Qwen2MoeForCausalLM}
REWARD_FN=$REPO/examples/python/math_reward_func.py
PROMPTS=${PROMPTS:-$REPO/tests/data/gsm8k_train_prompts.jsonl}
SFT_DATA=${SFT_DATA:-$REPO/tests/data/gsm8k_sft/train.parquet}
PREF_DATA=${PREF_DATA:-OpenRLHF/preference_dataset_mixture2_and_safe_pku}

FILTER=${1:-}
RESULTS=$REPO/tests/results/multigpu_extended_$(date +%Y%m%d_%H%M%S)
SUMMARY=$RESULTS/summary.txt
mkdir -p "$RESULTS"
PASS=0; FAIL=0; SKIP=0

export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1}
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=${OPENRLHF_DS_TORCH_ADAM:-1}   # no icpx -> no FusedAdam JIT
export OPENRLHF_WEIGHT_PROBE=0
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS/run.log"; }

if [[ ! -f "$PROMPTS" || ! -f "$SFT_DATA" ]]; then
    log "Generating E2E datasets..."
    "$PYTHON" "$REPO/tests/prepare_e2e_data.py" || exit 1
fi

ray_start() {  # ray_start [num_gpus]
    "$RAY" stop --force >/dev/null 2>&1; sleep 2
    "$RAY" start --head --num-gpus="${1:-2}" --disable-usage-stats >/dev/null 2>&1; sleep 4
}
ray_stop() { "$RAY" stop --force >/dev/null 2>&1; sleep 4; }

# ---------------------------------------------------------------------------
# Clear device state between cases. A straggler holding memory makes the NEXT
# case fail with a spurious OOM, which reads as a real result. Bracketed pkill
# patterns so the pattern cannot match this script's own command line.
# ---------------------------------------------------------------------------
IDLE_MIB=${IDLE_MIB:-600}
xpu_used() {
    local hi=0 v
    for d in 0 1; do
        v=$(timeout 12 xpu-smi stats -d "$d" 2>/dev/null \
            | grep "GPU Memory Used" | sed 's/.*current: //;s/[^0-9].*//')
        [[ "$v" =~ ^[0-9]+$ ]] && (( v > hi )) && hi=$v
    done
    echo "$hi"
}
clear_gpus() {
    "$RAY" stop --force >/dev/null 2>&1
    pkill -f "[r]ay::" 2>/dev/null
    pkill -f "[V]LLM::EngineCore" 2>/dev/null
    pkill -f "[o]penrlhf.cli" 2>/dev/null
    sleep 5
    local waited=0 used
    while (( waited < 60 )); do
        used=$(xpu_used)
        (( used < IDLE_MIB )) && return 0
        sleep 5; waited=$((waited+5))
    done
    log "  WARNING: XPUs still hold $(xpu_used) MiB — continuing"
}

# ---------------------------------------------------------------------------
# PER-CASE TIME BUDGETS. A case that exceeds its budget is KILLED and the suite
# continues -- one hang must never block the run.
#
# MEASURED 2026-09-19: a healthy RL case completes in ~270 s. These budgets are
# roughly 2x that, so "taking twice as long as expected" is the kill signal.
# `timeout` returns 124 on expiry, which record_* turns into TIMEOUT rather than a
# generic FAIL, so a hang is distinguishable from a crash in the root-cause pass.
# --kill-after sends SIGKILL if the process ignores SIGTERM.
# ---------------------------------------------------------------------------
T_RL=${T_RL:-600}            # RL case  (observed ~270 s)
T_SUP=${T_SUP:-600}          # supervised case
T_DET=${T_DET:-1300}         # determinism: two full RL runs
T_UNI=${T_UNI:-1500}         # universal ckpt: two RL runs + a conversion
T_LORA=${T_LORA:-900}        # lora_combiner: an SFT run + a merge
RUN() { timeout --kill-after=60 "$@"; }   # RUN <seconds> <cmd...>

skip_filtered() {
    [[ -z "$FILTER" || "$1" == *"$FILTER"* ]] && return 1
    log "SKIP  $1 — filtered"; echo "SKIP  $1 — filtered" >> "$SUMMARY"
    ((SKIP++)); return 0
}

# NOTE on `grep -c`: it prints 0 AND exits 1 when there are no matches, so
# `$(grep -c ... || echo 0)` yields the two-line string "0\n0" and breaks the
# numeric test. Every count below uses `|| true` plus a ${x:-0} default.
count() { local n; n=$(grep -cE "$2" "$1" 2>/dev/null || true); echo "${n:-0}"; }

record_rl() {   # record_rl <id> <desc> <rc> <logf>
    local id=$1 desc=$2 rc=$3 logf=$4
    local steps; steps=$(count "$logf" "Global step")
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        log "TIMEOUT $id — killed after its budget, steps=$steps"
        echo "FAIL  $id — $desc (TIMEOUT, steps=$steps)" >> "$SUMMARY"; ((FAIL++)); return
    fi
    if [[ $rc -eq 0 && ${steps:-0} -ge 1 ]]; then
        log "PASS  $id — $steps steps"
        echo "PASS  $id — $desc (steps=$steps)" >> "$SUMMARY"; ((PASS++))
    else
        local why; why=$(grep -oE "(AssertionError|RuntimeError|AttributeError|ValueError|TypeError|ImportError|OutOfMemoryError)[^\"]{0,110}" "$logf" 2>/dev/null | tail -1)
        log "FAIL  $id — rc=$rc steps=$steps"
        echo "FAIL  $id — $desc (rc=$rc, steps=$steps) ${why:+| $why}" >> "$SUMMARY"; ((FAIL++))
    fi
}

record_sup() {  # record_sup <id> <desc> <rc> <logf>
    local id=$1 desc=$2 rc=$3 logf=$4
    local n; n=$(count "$logf" "loss=|Loss:")
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        log "TIMEOUT $id — killed after its budget, loss_lines=$n"
        echo "FAIL  $id — $desc (TIMEOUT, loss_lines=$n)" >> "$SUMMARY"; ((FAIL++)); return
    fi
    if [[ $rc -eq 0 && ${n:-0} -ge 1 ]]; then
        log "PASS  $id — $n loss lines"
        echo "PASS  $id — $desc (loss_lines=$n)" >> "$SUMMARY"; ((PASS++))
    else
        local why; why=$(grep -oE "(AssertionError|RuntimeError|AttributeError|ValueError|TypeError|ImportError)[^\"]{0,110}" "$logf" 2>/dev/null | tail -1)
        log "FAIL  $id — rc=$rc loss_lines=$n"
        echo "FAIL  $id — $desc (rc=$rc, loss_lines=$n) ${why:+| $why}" >> "$SUMMARY"; ((FAIL++))
    fi
}

# ---------------------------------------------------------------------------
# run_rl  — GRPO base, remote reward FUNCTION, roles separated across 2 XPUs.
# Identical base config to the base suite's run_ppo_test so results compare.
# Extra flags come last, so they override any base value.
# ---------------------------------------------------------------------------
RL_BASE=(
  --actor.num_nodes 1 --actor.num_gpus_per_node 1
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager
  --vllm.sync_backend gloo
  --actor.model_name_or_path "$MODEL"
  --data.prompt_dataset "$PROMPTS"
  --data.input_key prompt --data.label_key label --data.apply_chat_template
  --data.max_len 512 --data.max_samples 40
  --train.batch_size 8 --train.micro_batch_size 2
  --rollout.batch_size 8 --rollout.max_new_tokens 256
  --train.max_epochs 1 --train.num_episodes 1
  --ds.zero_stage 2 --ds.adam_offload --ds.param_dtype bf16
  --ds.attn_implementation sdpa
  --algo.advantage.estimator group_norm --algo.kl.init_coef 0
  --rollout.n_samples_per_prompt 4
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1
)

# Any case that adds a role beyond actor+vLLM needs colocation. RL_BASE already
# uses both devices (actor 1 GPU + vLLM 1 engine), so a critic / ref / reward model
# would be a 3rd or 4th slot on a 2-GPU box and Ray's placement group can NEVER be
# satisfied -- it blocks forever rather than erroring. MEASURED: mg_lambda_gae hung
# 2h46m at the weight-sync init for exactly this reason. colocate_all + both sleeps
# is what upstream's train_ppo_ray_hybrid_engine.sh does and what the bench proves.
COLO=(--train.colocate_all --vllm.enable_sleep --ds.enable_sleep)

run_rl() {
    local id=$1 desc=$2; shift 2
    skip_filtered "$id" && return
    local logf=$RESULTS/$id.log
    log "START $id — $desc"
    clear_gpus; ray_start 2
    RUN "$T_RL" "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_BASE[@]}" \
        --reward.remote_url "$REWARD_FN" \
        --ckpt.output_dir /tmp/mgx_"$id" "$@" > "$logf" 2>&1
    local rc=$?
    ray_stop; record_rl "$id" "$desc" "$rc" "$logf"; clear_gpus
}

# run_rl_rm — same base but a reward MODEL, which is mutually exclusive with
# --reward.remote_url, so it cannot be expressed as extra flags on run_rl.
run_rl_rm() {
    local id=$1 desc=$2; shift 2
    skip_filtered "$id" && return
    local logf=$RESULTS/$id.log
    log "START $id — $desc"
    clear_gpus; ray_start 2
    RUN "$T_RL" "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_BASE[@]}" \
        --reward.model_name_or_path "$MODEL" \
        --reward.num_nodes 1 --reward.num_gpus_per_node 1 \
        --ckpt.output_dir /tmp/mgx_"$id" "$@" > "$logf" 2>&1
    local rc=$?
    ray_stop; record_rl "$id" "$desc" "$rc" "$logf"; clear_gpus
}

SUP_BASE=(
  --ds.zero_stage 2 --ds.adam_offload --ds.param_dtype bf16
  --ds.attn_implementation sdpa
  --train.max_epochs 1 --logger.logging_steps 1
  --eval.steps -1 --ckpt.save_steps -1
  --data.max_len 512 --data.max_samples 128
  --train.batch_size 8 --train.micro_batch_size 2
)

run_sup() {   # run_sup <id> <desc> <trainer> [flags...]
    local id=$1 desc=$2 trainer=$3; shift 3
    skip_filtered "$id" && return
    local logf=$RESULTS/$id.log
    log "START $id — $desc"
    clear_gpus
    RUN "$T_SUP" "$PYTHON" -m openrlhf.cli."$trainer" "${SUP_BASE[@]}" \
        --ckpt.output_dir /tmp/mgx_"$id" "$@" > "$logf" 2>&1
    local rc=$?
    record_sup "$id" "$desc" "$rc" "$logf"; clear_gpus
}

log "═══════════════════════════════════════════════════════════"
log "2-GPU EXTENDED suite — gap coverage (46 cases)"
log "repo=$REPO  results=$RESULTS"
log "═══════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# A. Advantage & policy-loss variants
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_no_std_norm "group_norm advantage WITHOUT std-dev normalization" \
  --algo.advantage.no_std_norm

# gamma is silently forced to 1.0 for every estimator except reinforce
# (experience_maker.py:308), so this case MUST use reinforce to mean anything.
run_rl mg_gamma_reinforce "Discount gamma=0.99 — only reinforce honours gamma" \
  --algo.advantage.estimator reinforce --algo.advantage.gamma 0.99

# lambd is a GAE parameter, so this needs a critic.
run_rl mg_lambda_gae "GAE lambda=0.95 (needs critic)" \
  --algo.advantage.estimator gae --algo.advantage.lambd 0.95 \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  "${COLO[@]}"

run_rl mg_gspo "GSPO policy loss instead of PPO surrogate" \
  --actor.policy_loss_type gspo

run_rl mg_dual_clip "Dual-clip PPO (lower bound on negative-advantage tokens)" \
  --actor.dual_clip 3.0


run_rl mg_value_clip "Critic value clipping (non-default)" \
  --algo.advantage.estimator gae --critic.value_clip 0.2 \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  "${COLO[@]}"

# ══════════════════════════════════════════════════════════════════════════
# B. KL control — both need a reference model, hence the ref role
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_kl_unbiased "Unbiased KL gradient on the KL-as-loss path" \
  --algo.kl.init_coef 0.01 --algo.kl.use_loss --algo.kl.estimator k3 \
  --algo.kl.unbiased_gradient \
  --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
  "${COLO[@]}"

run_rl mg_kl_adaptive "Adaptive KL controller (target + horizon)" \
  --algo.kl.init_coef 0.01 --algo.kl.target 0.01 --algo.kl.horizon 5000 \
  --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
  "${COLO[@]}"

# ══════════════════════════════════════════════════════════════════════════
# C. Importance-sampling correction — only the tv gate is untested
# ══════════════════════════════════════════════════════════════════════════
# tv gating is an UPPER BOUND only: a single delta, not a [low, high] band.
# MEASURED: two values -> "ValueError: is_correction_gating=tv takes an upper bound
# only (a single delta)". A band is valid only for ratio gating.
run_rl mg_is_tv_gating "IS correction gated by total-variation divergence" \
  --algo.advantage.is_correction_level token \
  --algo.advantage.is_correction_gating tv \
  --algo.advantage.is_correction_mode mask \
  --algo.advantage.is_correction_threshold 5e-3

# ══════════════════════════════════════════════════════════════════════════
# D. Reward & reference pipeline
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_reward_clip_range "Reward clipping to a narrow range" \
  --reward.clip_range -5 5

run_rl_rm mg_reward_model "Reward MODEL as a 4th resident model on 2 XPUs" \
  "${COLO[@]}"

run_rl_rm mg_reward_offload "Reward-model CPU offload" \
  --reward.offload \
  "${COLO[@]}"

run_rl_rm mg_colocate_critic_reward "colocate_critic_reward — critic+reward share a device" \
  --algo.advantage.estimator gae \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  --train.colocate_critic_reward \
  "${COLO[@]}"

run_rl mg_ref_offload "Reference-model CPU offload" \
  --algo.kl.init_coef 0.01 --ref.offload \
  --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
  "${COLO[@]}"

# ══════════════════════════════════════════════════════════════════════════
# E. vLLM rollout options
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_prefix_caching "vLLM prefix caching across samples of a prompt" \
  --vllm.enable_prefix_caching

# MEASURED (reproduced on retry): "ValueError: the new group's world size should be
# less or equal to the world size set by init_process_group". Retried here under
# colocate_all, which changes the placement and hence the collective group sizes.
# If it still fails, that constraint is the finding, not a flake.
# EXPECTED FAIL, recorded rather than worked around. Two configurations, two errors:
#   default placement -> ValueError: the new group's world size should be less or
#                        equal to the world size set by init_process_group
#   colocate_all      -> RuntimeError: The collective APIs shall be only used
#                        inside a Ray actor or task
# Ray's collective group has to be created inside a Ray actor, and on this path it
# is not. Not an XPU issue and not a config we can satisfy from the CLI; the
# --vllm.sync_with_ray path looks unmaintained. Kept as a tripwire.
run_rl mg_sync_with_ray "Weight sync via Ray (expected FAIL: collective needs Ray actor context)" \
  --vllm.sync_with_ray "${COLO[@]}"

run_rl mg_sampling_params "Non-default rollout temperature / top_p" \
  --rollout.temperature 0.7 --rollout.top_p 0.9

# ══════════════════════════════════════════════════════════════════════════
# F. DeepSpeed configuration
# ══════════════════════════════════════════════════════════════════════════
# The counterpart of our ZeRO-3 fix: overlap_comm explicitly ON at stage 2.
run_rl mg_overlap_comm "overlap_comm explicitly ON at ZeRO-2" \
  --ds.zero_stage 2 --ds.overlap_comm

run_rl mg_zpg "ZeRO++ hierarchical partitioning (zpg=2) at stage 3" \
  --ds.zero_stage 3 --ds.zpg 2

run_rl mg_grad_accum_dtype "Gradient-accumulation dtype forced to fp32" \
  --ds.grad_accum_dtype fp32

# EXPECTED TIMEOUT. MEASURED: hangs in ray::core::CoreWorker::Get(), i.e. the
# driver blocked on a remote call that never returns; the per-case timeout kills
# it. Kept because the hang itself is the finding.
run_rl mg_ppo_4bit "4-bit quantized actor in the RL loop (expected TIMEOUT: hangs in Ray get)" \
  --ds.load_in_4bit

# Isolates 4-bit from the rollout path: SFT uses no vLLM and no weight sync, so if
# this PASSES while mg_ppo_4bit hangs, quantization is fine and the RL/sync path is
# the problem. Added after the first run showed the RL case only as a timeout.
run_sup mg_sft_4bit "4-bit / QLoRA on SFT — isolates quantization from rollout" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --ds.load_in_4bit --ds.lora.rank 16 --ds.lora.alpha 32

# Gated on the import. MEASURED: "ModuleNotFoundError: No module named
# 'liger_kernel'" -- an absent OPTIONAL dependency is not a product failure, so
# this SKIPs rather than counting as a FAIL. `pip install liger-kernel` enables it.
if "$PYTHON" -c "import liger_kernel" 2>/dev/null; then
    run_rl mg_liger "Liger fused kernels" --ds.use_liger_kernel
else
    log "SKIP  mg_liger — liger_kernel not installed (optional dependency)"
    echo "SKIP  mg_liger — liger_kernel not installed (optional dependency)" >> "$SUMMARY"
    ((SKIP++))
fi

# EXPECTED LIKELY-FAIL: DeepCompile unproven on XPU.
run_rl mg_deepcompile "DeepSpeed DeepCompile" \
  --ds.deepcompile

# EXPECTED LIKELY-FAIL: no XPU flash-attn2 kernel published for torch 2.13.
run_rl mg_flash_attn2 "attn_implementation flash_attention_2" \
  --ds.attn_implementation flash_attention_2

# Actor spans BOTH XPUs, so vLLM must colocate with it.
# EXPECTED UNCERTAIN: DeepSpeed AutoTP on XPU is unproven.
# colocate_all asserts actor GPUs == engines x TP, so 2 actor GPUs needs 2 engines.
run_rl mg_ds_autotp2 "DeepSpeed AutoTP across both XPUs (ds.tensor_parallel_size 2)" \
  --actor.num_gpus_per_node 2 --ds.tensor_parallel_size 2 \
  --vllm.num_engines 2 \
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep

# Gated on the import. MEASURED: "ModuleNotFoundError: No module named
# 'ring_flash_attn'" -- an absent OPTIONAL dependency, exactly like liger_kernel,
# so this SKIPs rather than counting as a FAIL. The 2-engine flag is still correct
# (colocate_all asserts actor GPUs == engines x TP) for whenever the dep is present.
if "$PYTHON" -c "import ring_flash_attn" 2>/dev/null; then
    run_rl mg_ring_attn2 "Ring/sequence attention across 2 ranks (ring_attn_size 2)" \
      --actor.num_gpus_per_node 2 --ds.ring_attn_size 2 --ds.ring_attn_head_stride 1 \
      --vllm.num_engines 2 \
      --train.colocate_all --vllm.enable_sleep --ds.enable_sleep
else
    log "SKIP  mg_ring_attn2 — ring_flash_attn not installed (optional dependency)"
    echo "SKIP  mg_ring_attn2 — ring_flash_attn not installed (optional dependency)" >> "$SUMMARY"
    ((SKIP++))
fi

# ══════════════════════════════════════════════════════════════════════════
# F2. Mixture-of-Experts — a REAL MoE (granite-3.1-1b-a400m-instruct, 32 experts
#     top-8, 1.33B params). On a dense model --ds.experts_implementation and
#     aux_loss_coef are NO-OPS: no router to balance, no expert kernel to pick.
#
#     These run on the SUPERVISED path because MoE RL could not be validated on
#     this hardware. What was actually measured, 2026-09-19:
#
#     GraniteMoE  RL dies at the INITIAL broadcast_to_vllm, before any step:
#         vllm/model_executor/models/granitemoe.py:550 load_weights ->
#         KeyError: 'layers.0.block_sparse_moe.router.weight'
#       Cause is a NAME DIVERGENCE specific to this architecture. transformers 5.x
#       refactored GraniteMoE's runtime modules; the checkpoint and vLLM's loader
#       kept the original layout:
#         runtime     block_sparse_moe.router.weight / experts.gate_up_proj / experts.down_proj
#         checkpoint  block_sparse_moe.router.layer.weight / input_linear.weight / output_linear.weight
#       OpenRLHF sends transformers' runtime names straight to vLLM's load_weights
#       (which is the correct API -- it does NOT bypass the mapping). The two just
#       disagree for this architecture.
#
#     Qwen2Moe    NOT the same failure: the router KeyError is GONE and sync gets
#       past mlp.gate.weight and mlp.experts.gate_up_proj, because its runtime and
#       checkpoint names are identical. It then fails on a shape mismatch
#       (32 vs 64), which on a 2-layer/hidden-64 synthetic model is most likely an
#       artifact of the toy checkpoint rather than a defect.
#
#     CONCLUSION: this is NOT a general OpenRLHF MoE defect. It is an
#     architecture-specific name divergence, and MoE RL stays UNPROVEN here only
#     because no real small Qwen2Moe exists -- the smallest is Qwen1.5-MoE-A2.7B
#     at 14.3B params (~29 GB bf16), which will not fit on 2x16 GB alongside vLLM
#     and the optimizer. Not XPU-specific either way.
#
#     The supervised path needs no vLLM, so it exercises routers and expert
#     kernels for real. All four trainers accept both MoE flags.
# ══════════════════════════════════════════════════════════════════════════
MOE_SFT=(--model.model_name_or_path "$MOE_MODEL"
         --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template)

run_sup mg_moe_sft "Baseline SFT on a real MoE model" train_sft "${MOE_SFT[@]}"

# Runs on the TINY MoE, not granite. MEASURED directly on a bare forward pass:
#   granite-3.1-1b-a400m  aux_loss=int 0,  router_logits=None   <- BROKEN
#   tiny Qwen2Moe         aux_loss=Tensor, router_logits=present <- OK
#   tiny Mixtral          aux_loss=Tensor, router_logits=present <- OK
# transformers' GraniteMoE ignores output_router_logits even when it is set, and its
# OWN code then dies at modeling_granitemoe.py:644 on
# "aux_loss.to(loss.device) -> 'int' object has no attribute 'to'". So this is a
# transformers GraniteMoE defect, NOT an OpenRLHF one -- OpenRLHF's
# sft_trainer.py:202 .item() merely crashes on the same malformed output.
# (OpenRLHF could still be more defensive: its guard tests aux_loss_coef > 1e-8,
#  not whether the value is a tensor; rm_trainer.py:188 repeats the pattern.)
run_sup mg_moe_aux_loss "MoE router balancing aux loss (tiny MoE: granite's impl is broken)" train_sft \
  --model.model_name_or_path "$MOE_SMALL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --model.aux_loss_coef 0.001

run_sup mg_moe_experts_eager "MoE experts_implementation = eager" train_sft \
  "${MOE_SFT[@]}" --ds.experts_implementation eager

# batched_mm materialises down_proj[expert_ids]; peak memory scales with
# tokens x experts. MEASURED at the default size: OOM allocating 4.81 GiB at
# transformers/integrations/moe.py:156, after completing 4/64 steps -- so the
# kernel works and only the slice is too big. Smaller batch/seq, same code path.
# MEASURED twice: OOM at transformers/integrations/moe.py gathering
# gate_up_proj[expert_ids] (3.39 GiB) and down_proj[expert_ids] (4.81 GiB).
# batched_mm materialises one expert weight matrix per (token x top_k) pair, so on
# granite (32 experts, top-8, hidden 1024) the peak is INHERENT -- shrinking the
# batch moved the number but did not fix it. The tiny MoE (4 experts, hidden 64)
# exercises the same kernel selection within memory.
run_sup mg_moe_experts_batched_mm "MoE experts_implementation = batched_mm (tiny MoE)" train_sft \
  --model.model_name_or_path "$MOE_SMALL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --ds.experts_implementation batched_mm --train.micro_batch_size 1

# EXPECTED UNCERTAIN: grouped_mm needs a grouped-GEMM kernel for this backend.
run_sup mg_moe_experts_grouped_mm "MoE experts_implementation = grouped_mm" train_sft \
  "${MOE_SFT[@]}" --ds.experts_implementation grouped_mm

# EXPECTED FAIL: DeepGEMM is a CUDA-only kernel library.
run_sup mg_moe_experts_deepgemm "MoE experts_implementation = deepgemm" train_sft \
  "${MOE_SFT[@]}" --ds.experts_implementation deepgemm

run_sup mg_moe_rm "Reward-model training on a real MoE" train_rm \
  --model.model_name_or_path "$MOE_MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template

# TRIPWIRE, EXPECTED FAIL on GraniteMoE (runtime-vs-checkpoint name divergence,
# see the section header). Override MOE_MODEL to retest on another architecture:
#   MOE_MODEL=<a real small Qwen2Moe> bash ... mg_moe_grpo
# DEFERRED by decision 2026-09-19: MoE *RL* is parked until the MoE work is
# revisited separately. Two attempts, two different failures, zero steps:
# GraniteMoE dies on the router name divergence, tiny Qwen2Moe clears the names
# then fails on a shape mismatch that cannot be separated from a toy-checkpoint
# artifact. MoE SFT/RM below DO pass and stay in the run. Re-enable by
# uncommenting; MOE_MODEL is overridable.
# run_rl mg_moe_grpo "MoE RL rollout weight sync (expected FAIL: GraniteMoE name divergence)" \
#   --actor.model_name_or_path "$MOE_MODEL"

# ══════════════════════════════════════════════════════════════════════════
# G. Gradient checkpointing variant
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_grad_ckpt_reentrant "Reentrant gradient checkpointing (RL actor)" \
  --actor.gradient_checkpointing_enable --actor.gradient_checkpointing_reentrant

run_sup mg_sup_grad_ckpt_reentrant "Reentrant gradient checkpointing (SFT)" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --model.gradient_checkpointing_enable --model.gradient_checkpointing_reentrant

# ══════════════════════════════════════════════════════════════════════════
# H. Optimizers & adapters
# ══════════════════════════════════════════════════════════════════════════
# Muon is run with BOTH sleeps off on purpose: DeepSpeed's MuonWithAuxAdam fails
# the FusedAdam-only check in offload_states(), so ds_sleep would crash for a
# reason unrelated to Muon itself.
run_rl mg_muon_ppo "Muon optimizer on actor AND critic" \
  --algo.advantage.estimator gae \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  --actor.optim muon --critic.optim muon \
  "${COLO[@]}"

run_sup mg_muon_sft "Muon optimizer on SFT" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --optim muon

run_rl mg_lora_variants "LoRA with non-default alpha / dropout / target_modules" \
  --ds.lora.rank 8 --ds.lora.alpha 32 --ds.lora.dropout 0.05 \
  --ds.lora.target_modules q_proj v_proj

# ══════════════════════════════════════════════════════════════════════════
# I. Checkpoint lifecycle
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_ckpt_disable_ds "HF-only checkpointing (skip the DeepSpeed ckpt)" \
  --ckpt.save_steps 2 --ckpt.disable_ds --ckpt.save_hf \
  --ckpt.path /tmp/mgx_ckpt_disable_ds

run_rl mg_ckpt_best_metric "Best-checkpoint selection by metric + rotation" \
  --ckpt.save_steps 2 --ckpt.best_metric_key reward --ckpt.max_num 2 \
  --ckpt.path /tmp/mgx_ckpt_best

run_rl mg_critic_extras "Critic freezing + save value network" \
  --algo.advantage.estimator gae \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  --critic.freezing_steps 2 --critic.save_value_network \
  --ckpt.save_steps 3 --ckpt.path /tmp/mgx_critic_extras \
  "${COLO[@]}"

# ══════════════════════════════════════════════════════════════════════════
# J. Evaluation — the entire eval path is dark on 2 GPUs today
# ══════════════════════════════════════════════════════════════════════════
run_rl mg_ppo_eval "In-training eval: dataset + cadence + n_samples + temperature + split" \
  --eval.steps 2 --eval.dataset "$PROMPTS" --eval.split train \
  --eval.n_samples_per_prompt 2 --eval.temperature 0.8

run_sup mg_sft_eval "Eval on the supervised path" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --eval.steps 20

# ══════════════════════════════════════════════════════════════════════════
# K. Supervised trainer variants
# ══════════════════════════════════════════════════════════════════════════
run_sup mg_sft_zero3 "SFT at ZeRO-3 — real parameter sharding across 2 ranks" train_sft \
  --ds.zero_stage 3 \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template

# pretrain_mode hands the raw field straight to the tokenizer, so it needs a FLAT
# TEXT column. Our SFT parquet has only `messages` (an ndarray of dicts), which
# gives "ValueError: text input must be of type str". Flatten it once here.
PRETRAIN_DATA=/tmp/mgx_pretrain_text.parquet
if [[ ! -f "$PRETRAIN_DATA" ]]; then
    "$PYTHON" - <<'PYFLAT'
import pandas as pd
d = pd.read_parquet("/home/sdp/madhu/OpenRLHF-fresh/tests/data/gsm8k_sft/train.parquet")
d["text"] = ["\n".join(f"{t['role']}: {t['content']}" for t in m) for m in d["messages"]]
d[["text"]].to_parquet("/tmp/mgx_pretrain_text.parquet")
print("wrote /tmp/mgx_pretrain_text.parquet rows:", len(d))
PYFLAT
fi

run_sup mg_sft_pretrain_mode "SFT pretrain mode (plain LM loss, flat text column)" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$PRETRAIN_DATA" --data.input_key text \
  --model.pretrain_mode_enable

run_sup mg_rm_loss_type_fp32 "RM LogExp loss + fp32 loss computation" train_rm \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --model.loss_type logexp --model.compute_fp32_loss_enable

run_sup mg_rm_packing "RM sample packing at ZeRO-3" train_rm \
  --ds.zero_stage 3 --ds.packing_samples \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template

run_sup mg_dpo_nll "DPO with an auxiliary NLL regularizer" train_dpo \
  --model.model_name_or_path "$MODEL" --ref.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --model.beta 0.1 --model.nll_loss_coef 0.1

run_sup mg_dpo_lora "DPO + LoRA" train_dpo \
  --model.model_name_or_path "$MODEL" --ref.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --model.beta 0.1 --ds.lora.rank 16 --ds.lora.alpha 32

run_sup mg_dpo_packing "DPO sample packing" train_dpo \
  --ds.packing_samples \
  --model.model_name_or_path "$MODEL" --ref.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --model.beta 0.1

# ══════════════════════════════════════════════════════════════════════════
# L. Data handling
# ══════════════════════════════════════════════════════════════════════════
# MEASURED: a hand-written template with no assistant/completion boundary gives
# "ValueError: Training must contain at least one completion" -- the SFT trainer
# masks the prompt and needs to find where the completion starts. Pass the MODEL'S
# OWN template as the custom one: that still proves --data.tokenizer_chat_template
# is plumbed through, without inventing a template that cannot train.
CHAT_TMPL=$("$PYTHON" -c "
from transformers import AutoTokenizer
print(AutoTokenizer.from_pretrained('$MODEL').chat_template, end='')" 2>/dev/null)
# MEASURED progression: with my hand-written template -> "must contain at least one
# completion" (no assistant boundary). With the model's own template -> "must
# contain at least one complete gradient-accumulation window", a DIFFERENT error:
# the template is accepted and the shortfall is usable-sample count.
# sft_trainer.py:109 raises when len(dataloader) * epochs < accumulated_gradient,
# and accumulated_gradient = batch_size / micro_batch_size. Raising max_samples to 512
# was NOT enough because batch 4 / micro 1 still demands 4 surviving batches and this
# template drops most samples. Setting batch == micro_batch makes accumulation 1, so a
# single surviving batch is sufficient. That STILL failed, with an EMPTY dataloader,
# and the real cause was mine: sft_dataset.py:67-70 reads tokenizer_chat_template ONLY
# inside `if self.apply_chat_template:`. The case omitted --data.apply_chat_template, so
# the custom template was never applied and raw `messages` dicts reached the tokenizer,
# dropping every sample. The two flags are a pair; neither works alone.
run_sup mg_data_chat_template "Custom tokenizer chat template (model's own, passed explicitly)" train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --data.max_samples 512 --train.micro_batch_size 2 --train.batch_size 2 \
  --data.tokenizer_chat_template "$CHAT_TMPL"

# prompt_probs mixes two or more prompt datasets; the same file twice is a valid
# mixing test -- it exercises the weighting code without needing a second corpus.
run_rl mg_prompt_probs "Multi-dataset prompt mixing weights" \
  --data.prompt_dataset "$PROMPTS","$PROMPTS" --data.prompt_probs 0.5,0.5

# ══════════════════════════════════════════════════════════════════════════
# M. Correctness — determinism needs two runs compared, so it is hand-rolled
# ══════════════════════════════════════════════════════════════════════════
run_determinism() {
    local id=mg_determinism desc="Full determinism on RL (FINDING: diverges despite seed+full_determinism)"
    skip_filtered "$id" && return
    log "START $id — $desc"
    local a=$RESULTS/${id}_runA.log b=$RESULTS/${id}_runB.log
    for f in "$a" "$b"; do
        clear_gpus; ray_start 2
        RUN "$T_RL" "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_BASE[@]}" \
            --reward.remote_url "$REWARD_FN" \
            --train.full_determinism_enable --train.seed 1234 \
            --ckpt.output_dir /tmp/mgx_"$id" > "$f" 2>&1
        ray_stop
    done
    clear_gpus
    # Compare the FULL sequence, not the first 3: run A's first logged value was
    # exactly 0.0 while run B's was not, which is the signature of a misaligned
    # comparison rather than proof of non-determinism. Also report where they first
    # diverge so a real difference is readable.
    local la lb
    la=$(grep -oE "'policy_loss': -?[0-9.e-]+" "$a" | sed "s/.*: //" | tr '\n' ' ')
    lb=$(grep -oE "'policy_loss': -?[0-9.e-]+" "$b" | sed "s/.*: //" | tr '\n' ' ')
    {
      echo "runA policy_loss: $la"
      echo "runB policy_loss: $lb"
      awk -v A="$la" -v B="$lb" 'BEGIN{na=split(A,x," ");nb=split(B,y," ");
        if(na!=nb){printf("step COUNT differs: %d vs %d\n",na,nb)}
        for(i=1;i<=(na<nb?na:nb);i++) if(x[i]!=y[i]){printf("first divergence at step %d: %s vs %s\n",i,x[i],y[i]);exit}}'
    } >> "$RESULTS/mg_determinism_compare.txt"
    if [[ -n "$la" && "$la" == "$lb" ]]; then
        log "PASS  $id — identical losses"
        echo "PASS  $id — $desc (losses match: $la)" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — losses differ"
        echo "FAIL  $id — $desc | runA=[$la] runB=[$lb]" >> "$SUMMARY"; ((FAIL++))
    fi
}
run_determinism

# ---------------------------------------------------------------------------
# SFT twin of mg_determinism. No vLLM, no sampling -- so it separates "the compute
# kernels are non-deterministic" from "the rollout sampling is non-deterministic".
# See the header note on mg_determinism for the measured RL divergence.
# ---------------------------------------------------------------------------
run_determinism_sft() {
    local id=mg_determinism_sft
    local desc="Full determinism on SFT (no vLLM, no sampling) - isolates compute from rollout"
    skip_filtered "$id" && return
    log "START $id — $desc"
    local a=$RESULTS/${id}_runA.log b=$RESULTS/${id}_runB.log
    for f in "$a" "$b"; do
        clear_gpus
        RUN "$T_SUP" "$PYTHON" -m openrlhf.cli.train_sft "${SUP_BASE[@]}" \
            --model.model_name_or_path "$MODEL" \
            --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
            --train.full_determinism_enable --train.seed 1234 \
            --ckpt.output_dir /tmp/mgx_"$id" > "$f" 2>&1
    done
    clear_gpus
    local la lb
    la=$(grep -oE "loss=-?[0-9.]+" "$a" | head -8 | tr '\n' ' ')
    lb=$(grep -oE "loss=-?[0-9.]+" "$b" | head -8 | tr '\n' ' ')
    { echo "runA: $la"; echo "runB: $lb"; } >> "$RESULTS/mg_determinism_sft_compare.txt"
    if [[ -n "$la" && "$la" == "$lb" ]]; then
        log "PASS  $id — identical losses across two runs"
        echo "PASS  $id — $desc (losses match: $la)" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — losses differ"
        echo "FAIL  $id — $desc | runA=[$la] runB=[$lb]" >> "$SUMMARY"; ((FAIL++))
    fi
}
run_determinism_sft


# ══════════════════════════════════════════════════════════════════════════
# N. Universal checkpoint LOAD — closes the loop on ckpt_ds_zero_to_universal,
#    which only ever proved the CONVERSION, never that the output is loadable.
# ══════════════════════════════════════════════════════════════════════════
run_universal_ckpt_load() {
    local id=mg_universal_ckpt_load desc="Load a converted universal checkpoint"
    skip_filtered "$id" && return
    log "START $id — $desc"
    local logf=$RESULTS/$id.log ck=/tmp/mgx_uni_ckpt
    rm -rf "$ck"; clear_gpus; ray_start 2

    # 1. produce a real DeepSpeed ZeRO checkpoint
    RUN "$T_RL" "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_BASE[@]}" \
        --reward.remote_url "$REWARD_FN" \
        --ckpt.path "$ck" --ckpt.save_steps 2 \
        --ckpt.output_dir /tmp/mgx_"$id" > "$logf" 2>&1
    ray_stop

    # 2. convert it, exactly as examples/scripts/ckpt_ds_zero_to_universal.sh does
    local converted=0 d="$ck/_actor" tag=""
    if [[ -f "$d/latest" ]]; then
        tag=$(cat "$d/latest")
        echo "${tag}_uni" > "$d/latest_universal"
        "$PYTHON" -m deepspeed.checkpoint.ds_to_universal --inject_missing_state \
            --input_folder "$d/$tag" --output_folder "$d/${tag}_uni" >> "$logf" 2>&1
        [[ -n "$(ls -A "$d/${tag}_uni" 2>/dev/null)" ]] && converted=1
    fi

    # 3. the actual test: resume FROM the universal checkpoint
    local rc=1
    if (( converted )); then
        echo "--- resuming from universal checkpoint ---" >> "$logf"
        clear_gpus; ray_start 2
        RUN "$T_RL" "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_BASE[@]}" \
            --reward.remote_url "$REWARD_FN" \
            --ckpt.path "$ck" --ckpt.load_enable --ds.use_universal_ckpt \
            --ckpt.output_dir /tmp/mgx_"$id" >> "$logf" 2>&1
        rc=$?
        ray_stop
    else
        echo "conversion produced nothing -- cannot test the load path" >> "$logf"
    fi
    clear_gpus
    record_rl "$id" "$desc" "$rc" "$logf"
}
run_universal_ckpt_load

# ══════════════════════════════════════════════════════════════════════════
# O. LoRA combiner — an entire entrypoint no suite touches. Train a LoRA
#    adapter, then merge it back into the base model.
# ══════════════════════════════════════════════════════════════════════════
run_lora_combiner() {
    local id=mg_lora_combiner desc="lora_combiner: merge a trained adapter into the base model"
    skip_filtered "$id" && return
    log "START $id — $desc"
    local logf=$RESULTS/$id.log ad=/tmp/mgx_lora_adapter out=/tmp/mgx_lora_merged
    rm -rf "$ad" "$out"; clear_gpus

    RUN "$T_LORA" "$PYTHON" -m openrlhf.cli.train_sft "${SUP_BASE[@]}" \
        --model.model_name_or_path "$MODEL" \
        --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
        --ds.lora.rank 16 --ds.lora.alpha 32 \
        --ckpt.output_dir "$ad" > "$logf" 2>&1
    local train_rc=$?
    echo "--- adapter tree ---" >> "$logf"; find "$ad" -maxdepth 2 >> "$logf" 2>&1

    local rc=1
    if [[ $train_rc -eq 0 ]]; then
        echo "--- combining ---" >> "$logf"
        RUN 300 "$PYTHON" -m openrlhf.cli.lora_combiner \
            --model_path "$MODEL" --lora_path "$ad" --output_path "$out" >> "$logf" 2>&1
        rc=$?
    fi
    clear_gpus

    # PASS needs a real merged model on disk, not just exit 0.
    if [[ $rc -eq 0 && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
        log "PASS  $id — merged model written"
        echo "PASS  $id — $desc (files=$(find "$out" -type f | wc -l))" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — train_rc=$train_rc combine_rc=$rc"
        echo "FAIL  $id — $desc (train_rc=$train_rc, combine_rc=$rc)" >> "$SUMMARY"; ((FAIL++))
    fi
}
run_lora_combiner

log "═══════════════════════════════════════════════════════════"
log "DONE  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
log "summary: $SUMMARY"
log "═══════════════════════════════════════════════════════════"
[[ -f "$SUMMARY" ]] && cat "$SUMMARY"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
