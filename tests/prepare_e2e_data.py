#!/usr/bin/env python
"""Generate the small GSM8K-derived datasets the single-GPU E2E suites expect.

Writes (idempotent; pass --force to regenerate):
  tests/data/gsm8k_train_prompts.jsonl   400 prompts, {"prompt","label"}  (RL)
  tests/data/gsm8k_sft/train.parquet     256 rows, chat "messages" format (SFT)

RM/DPO data (OpenRLHF preference mixture) and the models are HF-hub
auto-downloads at run time and are NOT produced here.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROMPTS = os.path.join(DATA, "gsm8k_train_prompts.jsonl")
SFT_DIR = os.path.join(DATA, "gsm8k_sft")
SFT = os.path.join(SFT_DIR, "train.parquet")

SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if present")
    args = ap.parse_args()

    os.makedirs(SFT_DIR, exist_ok=True)

    if args.force or not os.path.exists(PROMPTS):
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train").select(range(400))
        with open(PROMPTS, "w") as f:
            for r in ds:
                label = r["answer"].split("####")[-1].strip().replace(",", "")
                f.write(json.dumps({"prompt": r["question"] + SUFFIX, "label": label}) + "\n")
        print(f"wrote {PROMPTS} (400 prompts)")
    else:
        print(f"exists {PROMPTS}")

    if args.force or not os.path.exists(SFT):
        from datasets import load_dataset
        import pandas as pd

        ds = load_dataset("openai/gsm8k", "main", split="train").select(range(256))
        rows = [
            {"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ]}
            for r in ds
        ]
        pd.DataFrame(rows).to_parquet(SFT)
        print(f"wrote {SFT} (256 rows)")
    else:
        print(f"exists {SFT}")


if __name__ == "__main__":
    main()
