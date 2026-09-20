# Gloo weight-sync validation matrix -- 20 scenarios x 4 DeepSpeed stages

- Run id: `run_20260919_093539_latest_full80_final`
- Progress: **80/80** executed
- Tally: PASS 63 / FAIL 9 / BLOCKED 8
- Backend: gloo only (CPU-staged actor -> vLLM weight broadcast); 2x Intel Arc Pro B70
- X5-X8 declared invalid and not run: critic world size can never exceed actor world size on a 2-XPU box (see `_X5_8_REASON` in run_gloo_matrix.py)

| ID | Actor WS | Critic WS | vLLM layout | Actor placement | vLLM placement | Colocation mode | Sleep | Backend | Cross-XPU transfer | Stage 0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X1 | 1 | 1 | 1 engine x TP1 | XPU 0 | XPU 1 | Actor/vLLM separated | Both On | Gloo | Yes | BLOCK (ASYNC) | BLOCK (ASYNC) | BLOCK (ASYNC) | BLOCK (ASYNC) |
| X2 | 1 | 1 | 1 engine x TP1 | XPU 0 | XPU 1 | Actor/vLLM separated | vLLM On / DS Off | Gloo | Yes | BLOCK (ASYNC) | BLOCK (ASYNC) | BLOCK (ASYNC) | BLOCK (ASYNC) |
| X3 | 1 | 1 | 1 engine x TP1 | XPU 0 | XPU 1 | Actor/vLLM separated | vLLM Off / DS On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X4 | 1 | 1 | 1 engine x TP1 | XPU 0 | XPU 1 | Actor/vLLM separated | Both Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X9 | 2 | 1 | 2 engines x TP1 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | One TP1 engine per XPU | Fully colocated | Both On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X10 | 2 | 1 | 2 engines x TP1 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | One TP1 engine per XPU | Fully colocated | vLLM On / DS Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X11 | 2 | 1 | 2 engines x TP1 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | One TP1 engine per XPU | Fully colocated | vLLM Off / DS On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X12 | 2 | 1 | 2 engines x TP1 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | One TP1 engine per XPU | Fully colocated | Both Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X13 | 2 | 1 | 1 engine x TP2 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | Both On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X14 | 2 | 1 | 1 engine x TP2 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | vLLM On / DS Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X15 | 2 | 1 | 1 engine x TP2 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | vLLM Off / DS On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X16 | 2 | 1 | 1 engine x TP2 | Actor rank 0 on XPU 0; rank 1 on XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | Both Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X17 | 2 | 2 | 2 engines x TP1 | Actor and Critic ranks span XPU 0 and XPU 1 | One TP1 engine per XPU | Fully colocated | Both On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X18 | 2 | 2 | 2 engines x TP1 | Actor and Critic ranks span XPU 0 and XPU 1 | One TP1 engine per XPU | Fully colocated | vLLM On / DS Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X19 | 2 | 2 | 2 engines x TP1 | Actor and Critic ranks span XPU 0 and XPU 1 | One TP1 engine per XPU | Fully colocated | vLLM Off / DS On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X20 | 2 | 2 | 2 engines x TP1 | Actor and Critic ranks span XPU 0 and XPU 1 | One TP1 engine per XPU | Fully colocated | Both Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X21 | 2 | 2 | 1 engine x TP2 | Actor and Critic ranks span XPU 0 and XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | Both On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X22 | 2 | 2 | 1 engine x TP2 | Actor and Critic ranks span XPU 0 and XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | vLLM On / DS Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X23 | 2 | 2 | 1 engine x TP2 | Actor and Critic ranks span XPU 0 and XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | vLLM Off / DS On | Gloo | Yes | FAIL 0/5 (A0) | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |
| X24 | 2 | 2 | 1 engine x TP2 | Actor and Critic ranks span XPU 0 and XPU 1 | TP rank 0 on XPU 0; rank 1 on XPU 1 | Fully colocated | Both Off | Gloo | Yes | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** | **PASS 5/5** |

## Code legend

| Code | Meaning |
|---|---|
| PASS 5/5 | All 13 pass criteria met, 5/5 steps, gloo weight sync verified fresh on both XPUs |
| BLOCK (ASYNC) | Rejected before launch: train_ppo_ray.py:686 asserts `not args.vllm.enable_sleep` under --train.async_enable |
| FAIL (A0) | Bug A at stage 0: AttributeError: 'FP16_UnfusedOptimizer' object has no attribute 'offload_states' |
| BLOCK (A3) | Bug A at stage 3: stage3.py:3285 AssertionError: Offloading is supported only for DeepSpeed FusedAdam |
| FAIL (HANG) | Actor finishes the first train epoch, then the driver blocks forever in ray::CoreWorker::Get(); killed at the no-progress cutoff. No weight-broadcast marker is ever reached |
| FAIL (PROF) | vLLM EngineCore aborted: `AssertionError: Error in memory profiling. Initial free memory X, current free memory Y` -- free memory GREW mid-profile because the colocated DeepSpeed engine released its offloaded state |
| FAIL (SEGV) | Fatal Python error: Segmentation fault in PolicyModelActor, no Python frame |
| FAIL (KV) | vLLM EngineCore refused to start: negative `Available KV cache memory` at gpu_memory_utilization=0.22 -- sizing limit, not a weight-sync defect |
| FAIL (?) | Failed with a signature not yet classified; see cases/<id>/train.log |
| - | Not yet run |
