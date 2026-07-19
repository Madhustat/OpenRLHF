# GRPO Timing Run — Qwen2.5-1.5B-Instruct on GSM8K

This documents the exact command to run a timed GRPO training pass, for comparing wall-clock
duration across hardware (Intel XPU vs. NVIDIA CUDA). The NVIDIA version of this run was
launched with GPU-pinning flags (`CUDA_VISIBLE_DEVICES`) specific to a shared/booked multi-GPU
machine — none of that applies on the single-GPU Intel XPU box, so this version omits it
entirely.

---

## Prerequisites — set up fresh on any new BMG machine

This assumes a fresh machine following `SETUP_new_machine_from_scratch.md` (Parts 0-10
already completed: `openrlhf_exp/OpenRLHF/venv-tf` environment working, `torch.xpu.is_available()`
confirmed `True`). Nothing below assumes any file already exists — everything is created here.

**1. Activate the environment and `cd` into the OpenRLHF checkout:**
```bash
cd ~/madhu/openrlhf_exp/OpenRLHF   # adjust to wherever your checkout actually is
source venv-tf/bin/activate
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
```

**2. Generate the GSM8K data files** (writes into the current directory — `OpenRLHF/` itself,
so the relative path used in the run command below resolves correctly):
```bash
python -c "
import json
from datasets import load_dataset

INSTRUCTION_SUFFIX = ' Please reason step by step, and put your final answer within \\\\boxed{}.'
ds = load_dataset('openai/gsm8k', 'main')

def extract_answer(ans):
    return ans.split('####')[-1].strip().replace(',', '')

with open('gsm8k_train_prompts.jsonl', 'w') as f:
    for row in ds['train'].select(range(400)):
        f.write(json.dumps({'prompt': row['question'] + INSTRUCTION_SUFFIX, 'label': extract_answer(row['answer'])}) + '\n')

with open('gsm8k_eval.jsonl', 'w') as f:
    for row in ds['test'].select(range(200)):
        f.write(json.dumps({'prompt': row['question'] + INSTRUCTION_SUFFIX, 'label': extract_answer(row['answer'])}) + '\n')
print('done')
"
```

**3. Confirm the reward function already exists** (it's part of the OpenRLHF repo itself, no
action needed — just verifying the path referenced below is correct on your checkout):
```bash
ls examples/python/math_reward_func.py
```

---

## Run command (Intel XPU — single GPU, no CUDA-specific pinning)

Run from inside the `OpenRLHF/` checkout directory (same directory Steps 1-3 above used), with
the venv and `PATH` already active:

```bash
ray stop --force && rm -rf /tmp/ray
NUM_GPUS=$(python -c "import torch; print(torch.xpu.device_count())")
ONEAPI_DEVICE_SELECTOR=level_zero:0 ray start --head --num-gpus="$NUM_GPUS"

time python -m openrlhf.cli.train_ppo_ray \
  --ref.num_nodes 1 --ref.num_gpus_per_node 1 \
  --actor.num_nodes 1 --actor.num_gpus_per_node 1 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 1 --vllm.sync_backend gloo \
  --vllm.gpu_memory_utilization 0.4 --vllm.enforce_eager \
  --train.colocate_all --train.colocate_actor_ref \
  --algo.advantage.estimator group_norm --algo.kl.init_coef 0 \
  --rollout.n_samples_per_prompt 4 --rollout.batch_size 8 \
  --actor.model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --reward.remote_url examples/python/math_reward_func.py \
  --data.prompt_dataset gsm8k_train_prompts.jsonl \
  --data.input_key prompt --data.label_key label --data.apply_chat_template \
  --data.max_samples 64 --data.max_len 512 \
  --train.batch_size 8 --train.micro_batch_size 4 \
  --ds.attn_implementation sdpa --ds.zero_stage 0 \
  --ckpt.output_dir /tmp/qwen15_timing_test_xpu --ckpt.save_steps -1 --logger.logging_steps 1 --eval.steps -1
```

`--data.prompt_dataset gsm8k_train_prompts.jsonl` is a relative path — it resolves against
whatever directory you're in when you run the command (Step 2 wrote the file there). If you
run this command from a different directory than where Step 2 ran, either `cd` back to that
directory first, or replace this with an absolute path to wherever the file actually is on
that machine.

### Why this differs from the NVIDIA version

| Setting | NVIDIA version | XPU version | Why |
|---|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Set (e.g. `2,3`) | **Omitted entirely** | NVIDIA-only env var; this box has exactly 1 XPU, nothing to pin/select among multiple GPUs |
| `ray start --num-gpus` | Not needed | **Required** (`ray start --head --num-gpus=$NUM_GPUS`) | Ray's GPU auto-detection is NVIDIA/CUDA-specific and registers 0 GPUs on XPU by default — see `SETUP_new_machine_from_scratch.md` Part 9. Skipping this causes the run to hang indefinitely in a placement-group wait, not a clean failure. |
| `--train.colocate_all --train.colocate_actor_ref` | Removed (7-GPU box, roles spread across separate GPUs) | **Kept** | Single-GPU machine — actor, reference model, and vLLM engine must all colocate on the one available device; there is no "spread across GPUs" option here. |
| `--vllm.sync_backend gloo` | Not set (defaults to `nccl`, which works natively on NVIDIA) | **Explicitly set to `gloo`** | On XPU, the default `nccl`-based weight-sync path silently no-ops (`PyNcclCommunicator` disables itself when no NCCL/RCCL is present) — this was a real bug found and fixed earlier in this project (see `PORTING_openrlhf_for_xpu.md`, `distributed_util.py` section). `gloo` is required for weight sync to actually happen. |
| `--vllm.gpu_memory_utilization` | Tuned per free memory on a *shared* machine (started at 0.85, reduced to 0.4 after hitting real contention from other jobs on the same GPUs) | `0.4` (same value, but for a different reason) | On XPU this single GPU is not shared with other jobs — 0.4 here is chosen simply because the actor + reference model + vLLM engine all colocate on the same 1 device (unlike NVIDIA's 2-GPU split), so vLLM must leave more headroom for the other two roles sharing that same GPU. |
| `--ds.attn_implementation sdpa` | Same | Same | Unrelated to hardware differences; kept identical for a fair timing comparison. |

### What to record from this run

- Total wall-clock time (`time` command's `real` output)
- Whether it completed without OOM/hang (if it does hit memory pressure, see the memory-sizing
  guidance in `SETUP_new_machine_from_scratch.md` — e.g. lowering `--vllm.gpu_memory_utilization`
  further, or reducing `--rollout.n_samples_per_prompt`/`--data.max_samples`)
- Any errors, to compare against the NVIDIA run's error history for this same command shape

### NVIDIA run — for comparison

The corresponding NVIDIA run used 2 pinned GPUs (`CUDA_VISIBLE_DEVICES=2,3`) on a shared
7-GPU machine, no `--train.colocate_all` (actor/reference/vLLM spread across the 2 pinned
GPUs instead of colocating), default `nccl` sync backend, and
`--vllm.gpu_memory_utilization` tuned down from an initial `0.85` to `0.4` after discovering
~30GB was already in use on both pinned GPUs by another job. Result pending at time of writing
this report — to be filled in once that run completes.
