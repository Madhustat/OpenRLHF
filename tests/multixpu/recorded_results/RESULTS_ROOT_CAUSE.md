# Failure root causes and retry outcomes

- Main run: `run_20260919_093539_latest_full80`
- Repo: `/home/sdp/madhu/OpenRLHF-fresh`
- oneCCL: `CCL_*=direct` ON (defaults fail at actor_ws=2 -- measured 2026-09-19)
- Final merged table: `run_20260919_093539_latest_full80_final/MATRIX_TABLE.md`

| Case | 1st | steps | Root cause | Retry | steps |
|---|---|---|---|---|---|
| X3-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X9-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X11-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X12-Z0 | FAIL | 0/5 | vLLM KV cache sized negative: gpu_memory_utilization too low for this topology. | **FAIL** | 0/5 |
| X13-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X14-Z0 | FAIL | 0/5 | Unclassified -- see train.log. | **PASS** | 5/5 |
| X15-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X17-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X17-Z1 | FAIL | 0/5 | Hang / case timeout with no crash signature. | **PASS** | 5/5 |
| X19-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X21-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X23-Z0 | FAIL | 0/5 | DeepSpeed sleep needs FusedAdam; torch AdamW in use (stage 0 / unfused optimizer path). | **FAIL** | 0/5 |
| X23-Z1 | FAIL | 0/5 | vLLM memory-profiling race during colocated init. | **PASS** | 5/5 |
| X24-Z0 | FAIL | 0/5 | vLLM KV cache sized negative: gpu_memory_utilization too low for this topology. | **PASS** | 5/5 |

**Passed on retry (order/transient): 4** X14-Z0, X17-Z1, X23-Z1, X24-Z0

**Still failing: 10** X3-Z0, X9-Z0, X11-Z0, X12-Z0, X13-Z0, X15-Z0, X17-Z0, X19-Z0, X21-Z0, X23-Z0
