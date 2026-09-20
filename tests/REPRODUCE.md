# Reproducing these tests after a clone

    git clone -b openrlfh_fresh_multixpu https://github.com/Madhustat/OpenRLHF.git
    cd OpenRLHF

All four suites default to the checkout they live in, so nothing needs editing.

## 1. Environment (the only thing NOT in this repo)

    VENV=/home/sdp/venvs/openrlhf-xccl-auto-detect-213
    export LD_LIBRARY_PATH="$VENV/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    export PYTHON=$VENV/bin/python
    export RAY=$VENV/bin/ray
    export PYTHONPATH=$PWD
    export OPENRLHF_DS_TORCH_ADAM=1

Three facts that are load-bearing on this hardware:
  * venv lib FIRST on LD_LIBRARY_PATH — otherwise torch 2.13 dies with
    "undefined symbol: urDeviceWaitExp" (base miniforge ships an older libur_loader)
  * OPENRLHF_DS_TORCH_ADAM=1 — no icpx here, so FusedAdam cannot JIT-build
  * CCL_*=direct stays ON for any 2-device run (run_gloo_matrix.py sets it). With
    oneCCL defaults every actor_world_size=2 case dies with UR_RESULT_ERROR_DEVICE_LOST.

Required versions: torch 2.13.0+xpu, vLLM 0.27.2.dev0+xpu (source build; 0.29 NOT
validated), DeepSpeed 0.19.1, Ray 2.55.0, transformers 5.16.0, Python 3.13.

## 2. Generate the datasets (once)

    $PYTHON tests/prepare_e2e_data.py

Writes tests/data/gsm8k_train_prompts.jsonl and tests/data/gsm8k_sft/train.parquet.
The matrix's prompt file is already committed at tests/multixpu/data/.

## 3. Run

    # 14 cases, ~50 min
    bash tests/multixpu/run_upstream_bench.sh
    bash tests/multixpu/run_upstream_bench.sh up_ppo_gae        # one case

    # 80 cases, ~5 h
    $PYTHON tests/multixpu/run_gloo_matrix.py --list            # print, run nothing
    $PYTHON tests/multixpu/run_gloo_matrix.py
    $PYTHON tests/multixpu/run_gloo_matrix.py --only X12 --stages 3

    # 57 cases, ~4.5 h
    bash tests/test_e2e_suite_multigpu_extended.sh
    bash tests/test_e2e_suite_multigpu_extended.sh mg_gspo      # one case

    # 28 cases, ~2.5 h
    bash tests/test_e2e_suite_multigpu.sh

    # any suite + automatic root-cause analysis and retry of failures
    $PYTHON tests/run_multigpu_extended_with_rca.py

## 4. Compare against what passed

    tests/multixpu/recorded_results/     the outcomes from 2026-09-20
    tests/MULTIXPU_TEST_PLAN.md          all 94 rows with expected result per case

Expected: 87 PASS, 3 FAIL, 2 SKIP, 1 N/A, 1 parked. The 3 failures and 2 skips are
expected and explained in MULTIXPU_TEST_PLAN.md — do not treat them as regressions.

## If a case hangs

The extended suite kills any case exceeding its budget (T_RL=600s vs a measured ~270s
healthy case) and moves on. The base suite has no such budget, so wrap it:

    timeout 900 bash tests/test_e2e_suite_multigpu.sh <case>

To clear stuck devices between attempts:

    bash tests/multixpu/helpers/cleanup_stuck.sh
