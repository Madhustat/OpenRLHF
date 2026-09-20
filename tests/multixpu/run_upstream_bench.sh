#!/usr/bin/env bash
# =============================================================================
# Upstream examples/scripts — adapted test bench for 2x Intel Arc Pro B70
# =============================================================================
#
# Every RL script in OpenRLHF's examples/scripts/ is authored for 4-8 GPUs and
# 4B-8B models. None runs unmodified on a 2-XPU box. This bench adapts each one
# to our hardware while PRESERVING its distinctive algorithm configuration, so a
# pass genuinely exercises that upstream recipe.
#
# Adaptations applied uniformly (hardware, not algorithm):
#   model            -> Qwen/Qwen2.5-0.5B          (upstream: Llama-3-8B / Qwen3-4B)
#   dataset          -> GSM8K math prompts         (upstream: prompt-collection / dapo-math-17k)
#   GPUs             -> 1 per role, 1 vLLM engine x TP1   (upstream: 8 GPUs, 4 engines x TP2)
#   seq/batch        -> max_len 512, batch 8       (upstream: 2048-74240, batch 128-1024)
#   attn             -> sdpa                       (upstream: flash_attention_2 — CUDA-only)
#   weight sync      -> gloo                       (upstream: nccl — CUDA-only)
#   ds.zero_stage    -> 2 for the RL cases         (upstream: 3). A memory/sharding mode
#                                                   change, not an algorithm change. The
#                                                   supervised cases keep upstream's stage
#                                                   verbatim (SFT 2, RM 3, DPO 3), so ZeRO-3
#                                                   is still covered. Flip with RL_ZERO=3.
#   ds.adam_offload  -> ON everywhere              (upstream: on in some scripts, commented
#                                                   out in train_reinforce_baseline_*)
#   FusedAdam        -> torch.optim.AdamW          (OPENRLHF_DS_TORCH_ADAM=1; no icpx on this
#                                                   box, so FusedAdam cannot JIT-build)
#   ds.tensor_parallel_size / ds.ring_attn_*       NOT dropped by us — they are commented out
#                                                   in the upstream scripts themselves.
#   packing          -> REMOVED                    (needs a torch-2.13 XPU flash-attn2 kernel,
#                                                   which kernels-community has not published)
#   max_new_tokens   -> 384                        (upstream 1024-64000; at 128 every GSM8K
#                                                   answer truncates, so every reward is 0)
#   reward.normalize -> REMOVED from the PPO cases (MEASURED: with a 0.5B model no GSM8K
#                                                   answer is correct, so reward variance is
#                                                   exactly 0 and dividing by std yields NaN in
#                                                   values/critic_loss/policy_loss. Verified:
#                                                   identical run without it gives finite losses
#                                                   critic 0.058, policy -0.008. A small-model
#                                                   scale artifact, not an XPU or code defect.)
#
# Algorithm flags are kept verbatim from upstream: advantage estimator, KL
# estimator/coefficient/use_loss, clip-higher bounds, dynamic filtering, gamma,
# entropy_coef, importance-sampling correction (is_correction_level/_mode/
# _gating/_threshold), dynamic batching, freeze_visual_encoder, learning rates, gradient
# checkpointing, ckpt.save_hf/load_enable/max_num.
#
# TWO algorithm-flag deviations, both forced and both noted at their case:
#
# 1. dynamic_filtering_range -> -0.1 1.0 (upstream 0 1), forced by model scale. The filter
#   keeps a prompt group only when min < avg_reward < max, strictly
#   (samples_generator.py:173). A 0.5B model never solves
#   a GSM8K problem, so every group scores avg_reward=0.00, which equals upstream's lower bound
#   and is therefore dropped -- the filter discards EVERY prompt and training does 0 steps.
#   That is dynamic filtering behaving exactly as designed (drop groups with no learning
#   signal), not a defect. Lowering the bound below 0 retains all-zero groups so the rest of
#   the pipeline is still exercised.
#
# 2. up_agent_async: kl.init_coef 0 (upstream 1e-5 + kl.use_loss + k2), forced by device
#   count. A non-zero KL coefficient needs a ref model; async_enable is incompatible with
#   colocate_all, so actor and the vLLM engine each hold a device exclusively and a ref role
#   would need a third GPU. Not reducible by a memory setting.
#
# PASS requires more than exit 0. Each case must show real training progress:
#   RL cases         >=1 "Global step" line AND a finite policy_loss
#   PPO+critic       additionally a finite critic_loss
#   Supervised       >=1 "loss=" line with a finite value
# Any NaN/Inf in a reported loss FAILS the case even if the process exits 0.
#
# Usage:
#   ./run_upstream_bench.sh              # all valid cases
#   ./run_upstream_bench.sh up_ppo_gae   # single case
#
# -----------------------------------------------------------------------------
# UPSTREAM SCRIPTS DELIBERATELY NOT ADAPTED
#   train_ppo_ray_slurm.sh          Slurm launcher; infrastructure, nothing XPU-specific
#   train_nonrl_slurm.sh            same
#   docker_run.sh                   container setup
#   nvidia_docker_install.sh        NVIDIA-only setup
# =============================================================================
set -uo pipefail

# Latest upstream, VERIFIED equal to github OpenRLHF/OpenRLHF main tip dc2a7ad3
# (2026-09-17), plus our changes. The old pre-merge tree is
# /home/sdp/madhu/wt-exp-multi-gloo, which is 40 commits behind.
# Default to the checkout this script lives in (tests/multixpu/ -> repo root) so a fresh
# clone reproduces against ITSELF rather than silently testing another tree.
SUITE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SUITE_DIR/../.." && pwd)}
VENV=${VENV:-/home/sdp/venvs/openrlhf-xccl-auto-detect-213}
PYTHON=$VENV/bin/python
RAY=$VENV/bin/ray

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
# SmolVLM-256M, not upstream's 2B-class VLM: MEASURED, Qwen2-VL-2B exhausts HOST RAM here
# (Ray killed the worker at 119/125 GB -- adam_offload keeps optimizer state in host memory,
# and a 2B model plus ref plus vLLM does not fit). 256M runs. Override with VLM_MODEL=...
VLM_MODEL=${VLM_MODEL:-HuggingFaceTB/SmolVLM-256M-Instruct}
# in-repo copy, committed alongside this script
PROMPTS=${PROMPTS:-$SUITE_DIR/data/gsm8k_train_prompts.jsonl}
SFT_DATA=${SFT_DATA:-$REPO/tests/data/gsm8k_sft/train.parquet}
PREF_DATA=${PREF_DATA:-OpenRLHF/preference_dataset_mixture2_and_safe_pku}
REWARD_FN=$REPO/examples/python/math_reward_func.py
AGENT_FN=$REPO/examples/python/agent_func.py

FILTER=${1:-}
TS=$(date +%Y%m%d_%H%M%S)
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results_$TS
SUMMARY=$OUT/summary.txt
mkdir -p "$OUT"
PASS=0; FAIL=0; SKIP=0

export LD_LIBRARY_PATH="$VENV/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
export OPENRLHF_DS_TORCH_ADAM=1          # no icpx on this box -> no FusedAdam JIT build
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

ray_start() { "$RAY" stop --force >/dev/null 2>&1; sleep 2
               "$RAY" start --head --num-gpus 2 --disable-usage-stats >/dev/null 2>&1; sleep 4; }
ray_stop()  { "$RAY" stop --force >/dev/null 2>&1; sleep 4; }

# ---------------------------------------------------------------------------
# Clear XPU state between cases. A leftover process holding device memory makes
# the NEXT case fail with a spurious OOM, which would be misread as a real
# result. Kill stragglers, then wait for both devices to actually drain rather
# than assuming they have.
#   - bracketed patterns ([r]ay) so pkill cannot match its own command line
#   - waits up to ~60s for memory to fall below IDLE_MIB, then reports and
#     continues (a stuck device is itself worth seeing in the log)
# ---------------------------------------------------------------------------
IDLE_MIB=${IDLE_MIB:-500}
RL_ZERO=${RL_ZERO:-2}   # ZeRO stage for the RL cases (upstream uses 3)

xpu_used() {  # highest per-device used MiB across both XPUs, 0 if unreadable
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
    pkill -f "[r]ay::"            2>/dev/null
    pkill -f "[V]LLM::EngineCore" 2>/dev/null
    pkill -f "[o]penrlhf.cli"     2>/dev/null
    sleep 5
    local waited=0 used
    while (( waited < 60 )); do
        used=$(xpu_used)
        (( used < IDLE_MIB )) && { log "  gpus clear (${used} MiB)"; return 0; }
        sleep 5; waited=$((waited+5))
    done
    log "  WARNING gpus still hold $(xpu_used) MiB after ${waited}s — continuing anyway"
}

# ---------------------------------------------------------------------------
# Metric extraction. "Global step N: {...}" carries a dict of real metrics; we
# pull individual keys out of it rather than trusting the exit code.
# ---------------------------------------------------------------------------
metric() {  # metric <logfile> <key>  -> last value, or empty
    grep -oE "'$2': -?[0-9]+\.?[0-9]*(e-?[0-9]+)?" "$1" 2>/dev/null | tail -1 | sed "s/.*: //"
}
finite() {  # finite <value> -> 0 if a real finite number
    [[ -n "$1" ]] || return 1
    [[ "$1" =~ ^-?[0-9]+\.?[0-9]*([eE]-?[0-9]+)?$ ]] || return 1
    return 0
}

record() {  # record <id> <desc> <rc> <logfile> <mode>
    local id=$1 desc=$2 rc=$3 logf=$4 mode=$5
    local steps ploss closs sloss detail="" ok=1

    if [[ "$mode" == "rl" || "$mode" == "rl_critic" ]]; then
        steps=$(grep -c "Global step" "$logf" 2>/dev/null || true); steps=${steps:-0}
        ploss=$(metric "$logf" policy_loss)
        detail="steps=$steps policy_loss=${ploss:-none}"
        [[ $rc -eq 0 ]] || ok=0
        [[ ${steps:-0} -ge 1 ]] || ok=0
        finite "$ploss" || ok=0
        if [[ "$mode" == "rl_critic" ]]; then
            closs=$(metric "$logf" critic_loss)
            detail="$detail critic_loss=${closs:-none}"
            finite "$closs" || ok=0
        fi
    else
        sloss=$(grep -oE "loss=-?[0-9]+\.?[0-9]*" "$logf" 2>/dev/null | tail -1 | sed 's/loss=//')
        local n; n=$(grep -c "loss=" "$logf" 2>/dev/null || true); n=${n:-0}
        detail="loss_lines=$n loss=${sloss:-none}"
        [[ $rc -eq 0 ]] || ok=0
        [[ ${n:-0} -ge 1 ]] || ok=0
        finite "$sloss" || ok=0
    fi

    if grep -qiE "'(policy_loss|critic_loss)': (nan|inf|-inf)" "$logf" 2>/dev/null; then
        ok=0; detail="$detail NON-FINITE-METRIC"
    fi

    if [[ $ok -eq 1 ]]; then
        log "PASS  $id — $detail"
        echo "PASS  $id — $desc | $detail" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — rc=$rc $detail"
        echo "FAIL  $id — $desc | rc=$rc $detail" >> "$SUMMARY"; ((FAIL++))
    fi
}

skip_if_filtered() {
    [[ -z "$FILTER" || "$1" == *"$FILTER"* ]] && return 1
    log "SKIP  $1 — filtered"; echo "SKIP  $1 — filtered" >> "$SUMMARY"; ((SKIP++)); return 0
}

run_rl() {  # run_rl <id> <desc> <mode:rl|rl_critic> [flags...]
    local id=$1 desc=$2 mode=$3; shift 3
    skip_if_filtered "$id" && return
    local logf=$OUT/$id.log
    log "START $id — $desc"
    clear_gpus                     # start from a known-clean device state
    ray_start
    "$PYTHON" -m openrlhf.cli.train_ppo_ray "$@" > "$logf" 2>&1
    local rc=$?
    ray_stop
    record "$id" "$desc" "$rc" "$logf" "$mode"
    clear_gpus                     # never leave memory held for the next case
}

run_sup() {  # run_sup <id> <desc> <trainer> [flags...]
    local id=$1 desc=$2 trainer=$3; shift 3
    skip_if_filtered "$id" && return
    local logf=$OUT/$id.log
    log "START $id — $desc"
    clear_gpus
    "$PYTHON" -m openrlhf.cli."$trainer" "$@" > "$logf" 2>&1
    local rc=$?
    record "$id" "$desc" "$rc" "$logf" "sup"
    clear_gpus
}

# Hardware envelope shared by every RL case (all roles on 2 XPUs, colocated+sleep).
RL_HW=(
  --actor.num_nodes 1 --actor.num_gpus_per_node 1
  # ref MUST match actor's node/GPU shape whenever colocation is on AND kl.init_coef > 0
  # (train_ppo_ray.py:46-52 asserts it). ref defaults to 8 GPUs, so omitting this fails
  # immediately for every case that uses a non-zero KL coefficient.
  --ref.num_nodes 1 --ref.num_gpus_per_node 1
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep
  --vllm.sync_backend gloo
  --actor.model_name_or_path "$MODEL"
  --data.prompt_dataset "$PROMPTS"
  --data.input_key prompt --data.label_key label --data.apply_chat_template
  --data.max_len 512 --data.max_samples 40
  --train.batch_size 8 --train.micro_batch_size 2
  --rollout.batch_size 8 --rollout.max_new_tokens 384
  --train.max_epochs 1 --train.num_episodes 1
  --ds.zero_stage "$RL_ZERO" --ds.adam_offload --ds.param_dtype bf16
  --ds.attn_implementation sdpa
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1
)

# ---------------------------------------------------------------------------
# serve_remote_rm.sh — the only case that is a server, not a training run, so it
# needs its own harness rather than run_rl/run_sup. Start the reward-model HTTP
# server, POST a real request, and require a finite numeric reward back. A
# process that merely starts listening is NOT a pass: the value head has to
# produce a number, which is what proves the forward pass ran on this hardware.
# Deviations from upstream: sdpa instead of flash_attention_2 (no XPU FA2 kernel
# for torch 2.13) and Qwen2.5-0.5B instead of an 8B RM. The value head is
# randomly initialised, so the reward is arbitrary — only finiteness is checked.
# ---------------------------------------------------------------------------
run_serve_rm() {
    local id=up_serve_rm desc="upstream serve_remote_rm — reward-model HTTP server"
    skip_if_filtered "$id" && return
    local logf=$OUT/$id.log port=${RM_PORT:-5017}
    log "START $id — $desc"
    clear_gpus

    # Runs upstream's FULL config: both devices visible (accelerate shards the model
    # across them) and --reward.normalize_enable on. That required fixing three real
    # upstream bugs, all reachable on CUDA, none XPU-specific:
    #   1. model.py gather: eos_indices on the primary device, values on the score head's
    #      device. Reported upstream as issue #938 (32B RM, cuda:7 vs cuda:0), CLOSED
    #      2025-04-07 as completed with the fix only in a comment, never merged.
    #   2. model.py mean/std: registered persistent=False, so device_map loading leaves
    #      them uninitialised -> std=0.0 -> reward=(r-0)/0=inf. Open upstream PR #1343.
    #   3. model.py normalisation: reward on the score head's device, mean/std buffers on
    #      the primary device (accelerate never dispatches them -- it warns "device_map
    #      keys do not match any submodules: ['mean','std']"). NOT reported upstream;
    #      found here because #938's one-line fix only moves the crash to the next line.
    # All three live in openrlhf/models/model.py. Without them this case must pin to one
    # device and drop --reward.normalize_enable.
    "$PYTHON" -m openrlhf.cli.serve_rm \
        --reward.model_name_or_path "$MODEL" \
        --host 127.0.0.1 --port "$port" \
        --ds.param_dtype bf16 --ds.attn_implementation sdpa \
        --reward.normalize_enable \
        --data.max_len 512 --batch_size 4 > "$logf" 2>&1 &
    local srv=$!

    # Wait for the port to accept connections; model load dominates startup.
    # Bail out early if the server process dies so a crash is not read as a timeout.
    local waited=0 up=0
    while (( waited < 300 )); do
        if "$PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; then
            up=1; break
        fi
        kill -0 $srv 2>/dev/null || break
        sleep 5; waited=$((waited+5))
    done

    local reward="none"
    if (( up )); then
        reward=$("$PYTHON" - <<RMCLIENT 2>>"$logf"
import json, urllib.request
body = json.dumps({"query": ["What is 2+2? The answer is 4."],
                   "prompts": ["What is 2+2?"]}).encode()
req = urllib.request.Request("http://127.0.0.1:$port/get_reward", data=body,
                             headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=180).read())
v = r["rewards"][0]
print(float(v[0]) if isinstance(v, list) else float(v))
RMCLIENT
)
        echo "client got reward=$reward" >> "$logf"
    fi

    kill $srv 2>/dev/null; wait $srv 2>/dev/null
    clear_gpus

    if finite "$reward"; then
        log "PASS  $id — reward=$reward"
        echo "PASS  $id — $desc | reward=$reward" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — server_up=$up reward=${reward:-none}"
        echo "FAIL  $id — $desc | server_up=$up reward=${reward:-none}" >> "$SUMMARY"; ((FAIL++))
    fi
}

# ---------------------------------------------------------------------------
# ckpt_ds_zero_to_universal.sh — needs an existing DeepSpeed ZeRO checkpoint as
# input, so this case produces one first: a short PPO run with --ckpt.save_steps
# writes _actor/ and _critic/, which is exactly the layout upstream's script
# special-cases. Then run upstream's own conversion command on it.
# PASS requires the <tag>_uni folder to contain files — an empty directory means
# the conversion bailed, and ds_to_universal can exit 0 having written nothing.
# ---------------------------------------------------------------------------
run_ckpt_universal() {
    local id=up_ckpt_universal desc="upstream ckpt_ds_zero_to_universal — ZeRO to universal"
    skip_if_filtered "$id" && return
    local logf=$OUT/$id.log ck=/tmp/up_ckpt_ds
    log "START $id — $desc"
    rm -rf "$ck"; clear_gpus
    ray_start

    # ckpt.save_steps is last on the line so it overrides RL_HW's -1 (argparse
    # keeps the final occurrence).
    "$PYTHON" -m openrlhf.cli.train_ppo_ray "${RL_HW[@]}" \
        --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
        --reward.remote_url "$REWARD_FN" \
        --algo.advantage.estimator gae --algo.kl.init_coef 0.01 \
        --rollout.n_samples_per_prompt 1 \
        --ckpt.output_dir /tmp/up_ckpt_hf \
        --ckpt.path "$ck" --ckpt.save_steps 2 > "$logf" 2>&1
    local rc=$?
    ray_stop; clear_gpus

    { echo "--- train rc=$rc ---"; echo "--- checkpoint tree ---"; find "$ck" -maxdepth 2; } >> "$logf" 2>&1

    # Replicate upstream's process_dir() for each of _actor/_critic.
    local converted=0 dirs=0
    for sub in _actor _critic; do
        local d="$ck/$sub"
        [[ -d "$d" ]] && dirs=$((dirs+1))
        [[ -f "$d/latest" ]] || continue
        local tag; tag=$(cat "$d/latest")
        echo "${tag}_uni" > "$d/latest_universal"
        echo "--- converting $d tag=$tag ---" >> "$logf"
        "$PYTHON" -m deepspeed.checkpoint.ds_to_universal --inject_missing_state \
            --input_folder "$d/$tag" --output_folder "$d/${tag}_uni" >> "$logf" 2>&1
        if [[ -d "$d/${tag}_uni" && -n "$(ls -A "$d/${tag}_uni" 2>/dev/null)" ]]; then
            converted=$((converted+1))
        fi
    done

    if (( converted > 0 )); then
        log "PASS  $id — converted $converted of $dirs checkpoint dir(s)"
        echo "PASS  $id — $desc | ckpt_dirs=$dirs converted=$converted" >> "$SUMMARY"; ((PASS++))
    else
        log "FAIL  $id — train rc=$rc, ckpt_dirs=$dirs, converted=0"
        echo "FAIL  $id — $desc | rc=$rc ckpt_dirs=$dirs converted=0" >> "$SUMMARY"; ((FAIL++))
    fi
}

log "═══════════════════════════════════════════════════════════"
log "Upstream examples/scripts bench — 2x Arc Pro B70"
log "model=$MODEL  dataset=GSM8K  results=$OUT"
log "═══════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# 1. train_ppo_ray_hybrid_engine.sh  -> PPO + critic (GAE), colocate_all+sleep
#    Upstream algo kept: gae, kl.init_coef 0.01, reward normalize, grad ckpt,
#    critic lr 9e-6, actor lr 5e-7. Dropped: --ds.packing_samples (no XPU kernel).
# ══════════════════════════════════════════════════════════════════════════
run_rl up_ppo_gae "upstream train_ppo_ray_hybrid_engine — PPO+critic GAE" rl_critic \
  "${RL_HW[@]}" \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  --reward.remote_url "$REWARD_FN" \
  --algo.advantage.estimator gae \
  --algo.kl.init_coef 0.01 \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode clip \
  --actor.adam.lr 5e-7 --critic.adam.lr 9e-6 \
  --rollout.n_samples_per_prompt 1 \
  --actor.gradient_checkpointing_enable \
  --train.dynamic_batch_enable --train.max_tokens_per_gpu 4096 \
  --ckpt.save_hf \
  --ckpt.output_dir /tmp/up_ppo_gae

# ══════════════════════════════════════════════════════════════════════════
# 2. train_ppo_with_reward_fn.sh  -> PPO + critic with a remote reward FUNCTION
#    Upstream uses --train.colocate_actor_ref (partial colocation). On 2 XPUs
#    with a critic that needs 3 exclusive devices, so colocate_all is used and
#    the difference is noted rather than silently dropped.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_ppo_reward_fn "upstream train_ppo_with_reward_fn — PPO+critic, remote reward fn" rl_critic \
  "${RL_HW[@]}" \
  --critic.num_nodes 1 --critic.num_gpus_per_node 1 \
  --reward.remote_url "$REWARD_FN" \
  --algo.advantage.estimator gae \
  --algo.kl.init_coef 0.01 \
  --actor.adam.lr 5e-7 --critic.adam.lr 9e-6 \
  --actor.gradient_checkpointing_enable \
  --rollout.micro_batch_size 2 \
  --ckpt.output_dir /tmp/up_ppo_reward_fn

# ══════════════════════════════════════════════════════════════════════════
# 3. train_reinforce_baseline_hybrid_engine.sh -> REINFORCE++ baseline
#    Upstream algo kept: reinforce_baseline, dynamic filtering, n_samples 8
#    (reduced to 4 for batch size 8). Dropped: agent_func (case 6 covers it),
#    packing, 64k context.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_reinforce_baseline "upstream train_reinforce_baseline_hybrid_engine — REINFORCE++" rl \
  "${RL_HW[@]}" \
  --reward.remote_url "$REWARD_FN" \
  --algo.advantage.estimator reinforce_baseline \
  --algo.kl.init_coef 1e-5 --algo.kl.use_loss --algo.kl.estimator k2 \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode mask \
  --actor.adam.lr 5e-7 --actor.entropy_coef 0.0 \
  --actor.gradient_checkpointing_enable \
  --rollout.n_samples_per_prompt 4 --rollout.micro_batch_size 2 \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.0 \
  --train.dynamic_batch_enable --train.max_tokens_per_gpu 4096 --rollout.max_tokens_per_gpu 8192 \
  --ckpt.save_hf --ckpt.max_num 3 \
  --ckpt.output_dir /tmp/up_reinforce_baseline

# ══════════════════════════════════════════════════════════════════════════
# 4. train_dapo_ray_hybrid_engine.sh -> DAPO
#    Upstream algo kept verbatim: group_norm, kl.use_loss + k3 + 1e-3,
#    clip-higher 0.2/0.27, dynamic filtering, n_samples 8 (-> 4).
#    is_correction_mode clip IS now available upstream (the old is_correction_type
#    was replaced by is_correction_level/_mode), so it is passed verbatim.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_dapo "upstream train_dapo_ray_hybrid_engine — DAPO" rl \
  "${RL_HW[@]}" \
  --reward.remote_url "$REWARD_FN" \
  --algo.advantage.estimator group_norm \
  --algo.kl.init_coef 1e-3 --algo.kl.use_loss --algo.kl.estimator k3 \
  --algo.advantage.gamma 1.0 \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode clip \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.adam.lr 5e-7 --actor.gradient_checkpointing_enable \
  --rollout.n_samples_per_prompt 4 --rollout.micro_batch_size 2 \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.0 \
  --ckpt.save_hf \
  --ckpt.output_dir /tmp/up_dapo

# ══════════════════════════════════════════════════════════════════════════
# 5. train_prorlv2_math_hybrid_engine.sh -> ProRL-v2
#    Upstream algo kept verbatim: reinforce_baseline, kl.use_loss + k2 + 1e-4,
#    clip-higher 0.2/0.27, dynamic filtering, n_samples 16 (-> 4).
# ══════════════════════════════════════════════════════════════════════════
run_rl up_prorlv2 "upstream train_prorlv2_math_hybrid_engine — ProRL-v2" rl \
  "${RL_HW[@]}" \
  --reward.remote_url "$REWARD_FN" \
  --algo.advantage.estimator reinforce_baseline \
  --algo.kl.init_coef 1e-4 --algo.kl.use_loss --algo.kl.estimator k2 \
  --algo.advantage.gamma 1.0 \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode mask \
  --algo.advantage.is_correction_threshold 0.5 5.0 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.adam.lr 1e-6 --actor.gradient_checkpointing_enable \
  --reward.stop_properly_penalty_coef 0.0 \
  --rollout.n_samples_per_prompt 4 \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.0 \
  --train.dynamic_batch_enable --train.max_tokens_per_gpu 4096 \
  --ckpt.save_hf \
  --ckpt.output_dir /tmp/up_prorlv2

# ══════════════════════════════════════════════════════════════════════════
# 6. train_reinforce_baseline_ray_agent_async.sh -> async + agent multi-turn
#    Async is INCOMPATIBLE with colocate_all and with vLLM sleep, so this case
#    cannot use RL_HW: actor gets XPU 0, the vLLM engine XPU 1, no colocation,
#    no sleep. This is the one case that genuinely needs both devices at once.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_agent_async "upstream train_reinforce_baseline_ray_agent_async — async + agent" rl \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
  --vllm.sync_backend gloo \
  --train.async_enable \
  --actor.model_name_or_path "$MODEL" \
  --train.agent_func_path "$AGENT_FN" \
  --data.prompt_dataset "$PROMPTS" \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_len 512 --data.max_samples 40 \
  --train.batch_size 8 --train.micro_batch_size 2 \
  --rollout.batch_size 8 --rollout.max_new_tokens 384 \
  --train.max_epochs 1 --train.num_episodes 1 \
  --ds.zero_stage "$RL_ZERO" --ds.adam_offload --ds.param_dtype bf16 \
  --ds.attn_implementation sdpa \
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1 \
  --algo.advantage.estimator reinforce_baseline \
  --train.partial_rollout_enable \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode mask \
  --actor.adam.lr 5e-7 --actor.entropy_coef 0.0 \
  --actor.gradient_checkpointing_enable \
  --algo.dynamic_filtering_enable --algo.dynamic_filtering_range -0.1 1.0 \
  --train.dynamic_batch_enable --train.max_tokens_per_gpu 4096 --rollout.max_tokens_per_gpu 8192 \
  --rollout.micro_batch_size 2 \
  --ckpt.save_hf --ckpt.max_num 3 \
  --rollout.n_samples_per_prompt 4 \
  --algo.kl.init_coef 0 \
  --ckpt.output_dir /tmp/up_agent_async
  # kl.init_coef 0 is the ONE forced algorithmic deviation here (upstream: 1e-5 + kl.use_loss
  # + k2). A non-zero KL coefficient requires a ref model, and async_enable is incompatible
  # with colocate_all, so actor and the vLLM engine each hold a device exclusively -- a ref
  # role would need a third GPU. Not reducible by a memory setting; it needs 3 devices.

# ══════════════════════════════════════════════════════════════════════════
# 7. train_vlm_math_hybrid_engine.sh -> VLM RL
#    SCOPE DISCOVERY. Qwen2-VL-2B is 4x the baseline model on a shared device;
#    OOM is a plausible and acceptable outcome. VLM forbids a critic and packing
#    (both hard asserts upstream); upstream's own estimator is reinforce_baseline,
#    which needs no critic, so it is kept verbatim.
#    max_len is 1024 here, not 512: max_prompt_length = max_len - max_new_tokens, and 512
#    left only 128 while a geometry3k prompt with image tokens needs ~145. Truncating is not
#    an option -- it breaks image-token/pixel_values alignment.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_vlm "upstream train_vlm_math_hybrid_engine — VLM RL (scope discovery)" rl \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.3 --vllm.enforce_eager \
  --train.colocate_all --vllm.enable_sleep --ds.enable_sleep \
  --vllm.sync_backend gloo \
  --actor.model_name_or_path "$VLM_MODEL" \
  --reward.remote_url "$REWARD_FN" \
  --data.prompt_dataset hiyouga/geometry3k \
  --data.input_key problem --data.label_key answer --data.apply_chat_template \
  --data.image_key images --data.max_images_per_prompt 1 \
  --actor.freeze_visual_encoder \
  --data.max_len 2048 --data.max_samples 20 \
  --train.batch_size 8 --train.micro_batch_size 1 \
  --rollout.batch_size 8 --rollout.micro_batch_size 1 --rollout.max_new_tokens 256 \
  --train.max_epochs 1 --train.num_episodes 1 \
  --ds.zero_stage "$RL_ZERO" --ds.adam_offload --ds.param_dtype bf16 \
  --ds.attn_implementation eager \
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.kl.init_coef 0 --algo.kl.use_loss --algo.kl.estimator k2 \
  --algo.advantage.gamma 1.0 \
  --actor.adam.lr 2e-6 --actor.gradient_checkpointing_enable \
  --rollout.n_samples_per_prompt 4 \
  --ckpt.save_hf \
  --ckpt.output_dir /tmp/up_vlm

# ══════════════════════════════════════════════════════════════════════════
# 8-11. Supervised scripts. Single GPU by design upstream; only model, dataset
#       and lengths are scaled. zero_stage kept as upstream sets it (2 for SFT,
#       3 for RM/DPO). Packing dropped for the same flash-attn kernel reason.
# ══════════════════════════════════════════════════════════════════════════
SUP_HW=(--ds.adam_offload --ds.param_dtype bf16 --ds.attn_implementation sdpa
        --train.max_epochs 1 --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1
        --data.max_len 512 --data.max_samples 128
        --train.batch_size 8 --train.micro_batch_size 2)

# 8. train_sft.sh
run_sup up_sft "upstream train_sft — SFT full fine-tune (ZeRO-2)" train_sft \
  "${SUP_HW[@]}" --ds.zero_stage 2 \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --adam.lr 5e-6 --model.gradient_checkpointing_enable --ckpt.load_enable \
  --ckpt.output_dir /tmp/up_sft

# 9. train_sft_mixtral_lora.sh  (LoRA variant; upstream targets Mixtral)
run_sup up_sft_lora "upstream train_sft_mixtral_lora — SFT + LoRA" train_sft \
  "${SUP_HW[@]}" --ds.zero_stage 2 \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$SFT_DATA" --data.input_key messages --data.apply_chat_template \
  --ds.lora.rank 16 --ds.lora.alpha 32 \
  --adam.lr 5e-6 --model.gradient_checkpointing_enable --ckpt.load_enable \
  --ckpt.output_dir /tmp/up_sft_lora

# 10. train_rm.sh
run_sup up_rm "upstream train_rm — Reward Model (ZeRO-3)" train_rm \
  "${SUP_HW[@]}" --ds.zero_stage 3 \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --adam.lr 9e-6 --model.gradient_checkpointing_enable --ckpt.load_enable \
  --train.micro_batch_size 1 \
  --ckpt.output_dir /tmp/up_rm

# 11. train_dpo_llama.sh
run_sup up_dpo "upstream train_dpo_llama — DPO beta=0.1 (ZeRO-3)" train_dpo \
  "${SUP_HW[@]}" --ds.zero_stage 3 \
  --model.model_name_or_path "$MODEL" --ref.model_name_or_path "$MODEL" \
  --data.dataset "$PREF_DATA" \
  --data.chosen_key chosen --data.rejected_key rejected --data.apply_chat_template \
  --model.beta 0.1 \
  --adam.lr 5e-7 --model.gradient_checkpointing_enable --ckpt.load_enable \
  --train.micro_batch_size 1 \
  --ckpt.output_dir /tmp/up_dpo

# ══════════════════════════════════════════════════════════════════════════
# 14. train_flash_reinforce_ray_agent_async.sh -> FlashREINFORCE  (NEW upstream,
#     added after our previous base; did not exist when this bench was written)
#     Critic-free, single-rollout async RL. Upstream's header states the algorithm
#     IS the configuration, so these are kept verbatim:
#       estimator flash_reinforce, is_correction_level seq + gating binary_kl
#       + threshold 5e-3 (per-sequence trust region), loss_agg_mode
#       seq-mean-token-mean, n_samples_per_prompt 1, kl.init_coef 0.
#     LOAD-BEARING INVARIANT: train.batch_size MUST equal rollout.batch_size and
#     max_epochs must be 1, otherwise the PPO ratio is not 1 and this stops being
#     FlashREINFORCE. Both are 8/8 and 1 here, so the invariant holds at our scale.
#     Like up_agent_async, async_enable forbids colocate_all, so actor takes one
#     XPU and the vLLM engine the other. kl.init_coef 0 is upstream's own value
#     here, so unlike up_agent_async this case has NO forced deviation.
# ══════════════════════════════════════════════════════════════════════════
run_rl up_flash_reinforce "upstream train_flash_reinforce_ray_agent_async — FlashREINFORCE" rl \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 \
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
  --vllm.sync_backend gloo \
  --train.async_enable --train.async_queue_size 8 \
  --actor.model_name_or_path "$MODEL" \
  --reward.remote_url "$REWARD_FN" \
  --data.prompt_dataset "$PROMPTS" \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_len 512 --data.max_samples 40 \
  --train.batch_size 8 --train.micro_batch_size 1 \
  --rollout.batch_size 8 --rollout.micro_batch_size 1 \
  --rollout.vllm_generate_batch_size 16 \
  --rollout.max_new_tokens 384 --rollout.temperature 1.0 --rollout.top_p 1.0 \
  --rollout.n_samples_per_prompt 1 \
  --train.max_epochs 1 --train.num_episodes 1 \
  --ds.zero_stage "$RL_ZERO" --ds.adam_offload --ds.param_dtype bf16 \
  --ds.attn_implementation sdpa \
  --actor.gradient_checkpointing_enable \
  --actor.adam.lr 1e-6 --actor.adam.weight_decay 0.1 \
  --algo.advantage.estimator flash_reinforce \
  --algo.advantage.is_correction_level seq \
  --algo.advantage.is_correction_gating binary_kl \
  --algo.advantage.is_correction_threshold 5e-3 \
  --actor.loss_agg_mode seq-mean-token-mean \
  --algo.kl.init_coef 0 \
  --logger.logging_steps 1 --eval.steps -1 --ckpt.save_steps -1 \
  --ckpt.output_dir /tmp/up_flash_reinforce

# 12. serve_remote_rm.sh
run_serve_rm

# 13. ckpt_ds_zero_to_universal.sh
run_ckpt_universal

log "═══════════════════════════════════════════════════════════"
log "DONE  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
log "summary: $SUMMARY"
log "═══════════════════════════════════════════════════════════"
[[ -f "$SUMMARY" ]] && cat "$SUMMARY"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
