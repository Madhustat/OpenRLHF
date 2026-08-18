#!/usr/bin/env python
"""Generate the E2E-suite datasets from GSM8K if they are missing.

The E2E suites need two small local datasets that are not committed:

  tests/data/gsm8k_train_prompts.jsonl   RL prompts  ({"prompt","label"})
  tests/data/gsm8k_sft/train.parquet     SFT data    ({"messages"})

Both are derived from openai/gsm8k (main). This script is idempotent: it
skips files that already exist. Pass --force to regenerate.

Usage:
    python tests/prepare_e2e_data.py            # generate if missing
    python tests/prepare_e2e_data.py --force    # regenerate
"""
import argparse
import json
import os

# RL prompts get a fixed suffix so the math reward fn can parse \boxed{} answers.
PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
N_RL_PROMPTS = 400   # RL suite caps samples well below this; extra is harmless
N_SFT_SAMPLES = 512  # SFT tests use --data.max_samples 128; extra is harmless


def gsm8k_label(answer: str) -> str:
    """GSM8K answers end with '#### <number>'; return the number as the label."""
    return answer.split("####")[-1].strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if present")
    args = ap.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    prompts_path = os.path.join(data_dir, "gsm8k_train_prompts.jsonl")
    sft_dir = os.path.join(data_dir, "gsm8k_sft")
    sft_path = os.path.join(sft_dir, "train.parquet")
    os.makedirs(sft_dir, exist_ok=True)

    have_prompts = os.path.exists(prompts_path)
    have_sft = os.path.exists(sft_path)
    if have_prompts and have_sft and not args.force:
        print(f"E2E data already present under {data_dir} (use --force to regenerate)")
        return

    from datasets import load_dataset

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    if not have_prompts or args.force:
        with open(prompts_path, "w") as f:
            for row in gsm8k.select(range(N_RL_PROMPTS)):
                rec = {"prompt": row["question"] + PROMPT_SUFFIX, "label": gsm8k_label(row["answer"])}
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {N_RL_PROMPTS} RL prompts -> {prompts_path}")

    if not have_sft or args.force:
        import pandas as pd

        rows = []
        for row in gsm8k.select(range(N_SFT_SAMPLES)):
            rows.append({"messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ]})
        pd.DataFrame(rows).to_parquet(sft_path)
        print(f"wrote {N_SFT_SAMPLES} SFT samples -> {sft_path}")


if __name__ == "__main__":
    main()
