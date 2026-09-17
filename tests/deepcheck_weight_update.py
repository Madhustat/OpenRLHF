#!/usr/bin/env python
"""Deep-check #3 (run-level): prove RL training actually UPDATED the weights.

Loads the base model and a trained/exported HF model, compares their parameter
tensors, and asserts at least one trainable parameter changed (and reports how
many). A run that "passed" 10 steps but never optimized would show 0 changed.

Usage:
    python tests/deepcheck_weight_update.py <base_model> <trained_hf_dir>
Exit 0 = weights changed (PASS); exit 1 = unchanged or error (FAIL).
"""
import sys

import torch
from transformers import AutoModelForCausalLM


def main():
    if len(sys.argv) != 3:
        print("usage: deepcheck_weight_update.py <base_model> <trained_hf_dir>")
        return 2
    base_id, trained_dir = sys.argv[1], sys.argv[2]

    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    trained = AutoModelForCausalLM.from_pretrained(trained_dir, torch_dtype=torch.float32)

    bsd, tsd = base.state_dict(), trained.state_dict()
    shared = [k for k in bsd if k in tsd and bsd[k].shape == tsd[k].shape]
    assert shared, "no comparable parameters between base and trained model"

    changed, total_delta = 0, 0.0
    for k in shared:
        d = (bsd[k].float() - tsd[k].float()).abs().max().item()
        if d > 1e-6:
            changed += 1
            total_delta += d

    print(f"DEEPCHECK-WEIGHTUPDATE: {changed}/{len(shared)} params changed, "
          f"summed max-abs-delta={total_delta:.4f}")
    if changed == 0:
        print("DEEPCHECK-WEIGHTUPDATE-VIOLATION: no parameter changed — training did not optimize")
        return 1
    print("DEEPCHECK-WEIGHTUPDATE OK: weights genuinely updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
