#!/bin/bash
# Phase 9: checkpoint resume validation.
# Run A: train 2 epochs, save DS checkpoint each epoch. Run B: load_enable -> resume, continue.
set -o pipefail
LOGFILE="$1"
BASE=/tmp/ckpt_resume_$(basename "$LOGFILE" .log)
CKPT_OUT=$BASE/output
CKPT_PATH=$BASE/checkpoints_sft

cd /home/dut7054/openrlfh_exp/OpenRLHF
source /home/dut7054/madhu/work2_with_transformers_native/OpenRLHF/venv-tf/bin/activate
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
COMMON="--data.max_len 512 --data.dataset Open-Orca/OpenOrca --data.input_key question \
  --data.output_key response --data.max_samples 12 --train.batch_size 2 --train.micro_batch_size 1 \
  --model.model_name_or_path $MODEL --ckpt.output_dir $CKPT_OUT --ckpt.path $CKPT_PATH \
  --ckpt.save_steps 5 --logger.logging_steps 1 --eval.steps -1 --ds.zero_stage 0 \
  --ds.param_dtype bf16 --ds.attn_implementation sdpa --adam.lr 5e-6"

echo "########## RUN A: initial train (2 epochs), save checkpoint ##########" | tee -a "$LOGFILE"
python -m openrlhf.cli.train_sft $COMMON --train.max_epochs 2 >>"$LOGFILE" 2>&1
echo "RUN_A_exit=$?" | tee -a "$LOGFILE"
echo "=== checkpoint dir after Run A ===" | tee -a "$LOGFILE"
ls -1 "$CKPT_PATH" 2>&1 | tee -a "$LOGFILE"
echo "=== latest tag ===" | tee -a "$LOGFILE"
cat "$CKPT_PATH/latest" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "########## RUN B: resume from checkpoint (--ckpt.load_enable), continue to epoch 3 ##########" | tee -a "$LOGFILE"
python -m openrlhf.cli.train_sft $COMMON --train.max_epochs 3 --ckpt.load_enable >>"$LOGFILE" 2>&1
echo "RUN_B_exit=$?" | tee -a "$LOGFILE"
