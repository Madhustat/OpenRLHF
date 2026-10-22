# Weight sync to vLLM on Intel XPU

After each training step, OpenRLHF broadcasts the updated weights from DeepSpeed rank 0 to the vLLM
engines. On Intel XPU (Intel Arc / Data Center GPUs), two backends exist:

| Backend | Transfer | When |
|---|---|---|
| `xccl` | direct XPU-to-XPU over oneCCL (ProcessGroupXCCL), no CPU staging | preferred, when every prerequisite below holds |
| `gloo` | CPU-staged broadcast | fallback, single-XPU layouts, unsupported environments |

`--vllm.sync_backend` selects it: `auto` (default), `xccl`, `gloo` (or `nccl` on CUDA/ROCm).
NVIDIA and AMD behave as before: `auto` selects `nccl`.

## Pinning without masking

Each vLLM engine must run on the XPU Ray assigned it. oneCCL needs **every XPU of a communicator
visible in every participating process**: a masked vLLM engine makes XCCL hang, or with some oneCCL
settings return wrong data, without an error. So vLLM is pinned by device index, with either backend:
- no `ZE_AFFINITY_MASK` and no restricting `ONEAPI_DEVICE_SELECTOR` for the vLLM actor;
- the Ray-assigned XPU IDs are passed to vLLM's `device_ids` engine argument;
- the trainer selects its own XPU by index, as it already does with
  `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1`.

This needs **vLLM >= 0.31** on XPU: older XPU workers ignore `device_ids` and bind `xpu:{local_rank}`
(fixed by vllm-project/vllm#56015). OpenRLHF checks at startup that the installed XPU worker places
itself by `device_ids` and refuses to start otherwise; it never falls back to placement by local rank.

Set `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` on XPU **with either backend**. Without it,
Ray sets `ONEAPI_DEVICE_SELECTOR` only after an actor's XPU runtime has started, so the trainer still
sees every XPU and binds xpu:0, whatever XPU Ray assigned it.

## XCCL eligibility checks

Six **hard prerequisites** of the environment, required by `auto` and by an explicit `xccl`:
1. A PyTorch XPU build with XCCL.
2. PyTorch >= 2.13 (2.12's XCCL broadcast crashes, intel/torch-xpu-ops#4238).
3. `RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1` exported **before** `ray start`, so Ray does not
   restrict each actor to one XPU.
4. No oneCCL setting known to break it: `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` (GPU page faults on large
   tensors), `FI_PROVIDER=shm` (hang), `CCL_TOPO_ALGO=0` (crash).
5. vLLM >= 0.31 (validated with 0.31.0).
6. `pidfd_getfd()` permitted: oneCCL shares XPU memory between processes with it. In Docker, run with
   `--cap-add SYS_PTRACE` (or `--security-opt seccomp=unconfined`); Docker's default seccomp profile
   denies the call, and oneCCL's fallback can hang (uxlfoundation/oneCCL#217).

Two **auto-selection topology policies**: `auto` selects gloo for
1. colocated execution (`--train.colocate_all`, vLLM shares XPUs with the trainer), and
2. a single XPU,

because gloo is the currently validated default there. An explicit `xccl` remains available as an
experimental option for colocated execution; without `colocate_all` it needs two XPUs.

`auto` runs all of these checks before any actor is created and uses gloo, logging the reason, if one
fails. An explicit `xccl` stops with the reason of a failed hard prerequisite instead.

## Safety checks

Before the first real weight, the trainer checks every rank of the weight-sync communicator by
physical identity (device UUID and PCI address, never only `xpu:N`):
- each rank runs on the XPU Ray assigned it;
- no rank holds memory on another XPU (bitsandbytes' import-time lookup tables are the one
  tolerated exception: transformers imports bitsandbytes before an actor can select its XPU);
- every rank can see every communicator XPU;
- no two ranks share an XPU unless vLLM is colocated.

Then rank 0 broadcasts deterministic 4 KiB, 16 MiB and 128 MiB tensors into poison-filled buffers,
and every rank checks them bit-exact. Failing ranks are reported with UUID, PCI address, size and
phase.

ProcessGroupXCCL reports broadcasts complete before the device work is done, and its timeout does
not fire, so a hung XCCL collective would block forever. Every real XCCL weight sync therefore runs
under a watchdog (`--vllm.sync_timeout`, default 600 s per tensor); on timeout it reports the tensor,
phase and ranks and stops the job.

## Failure behavior

| Situation | Result |
|---|---|
| `auto`, an eligibility check fails | gloo from the start, reason logged |
| `auto`, placement check or preflight fails during setup, or the setup watchdog ends a hung preflight | every vLLM and training actor is stopped and recreated with gloo (pinned by the same `device_ids`), placement is re-checked, training starts |
| `auto`, any other actor death during setup (OOM, model load, invalid configuration, crash) | the job stops with the original error; no fallback |
| explicit `xccl`, any check fails | the job stops with the reason; no fallback |
| explicit `gloo` | gloo; no XCCL group or preflight (the placement check still runs) |
| XCCL fails after training started | the job stops (watchdog); a process that touched XCCL is never reused, and nothing falls back mid-training |

## Not supported on XPU

- vLLM pipeline parallelism: rejected at startup. Its stage-to-stage sends can stop completing while
  vLLM's sampled-token broadcast runs on a second XCCL communicator between the same ranks.
- vLLM data parallelism.

Docker example for the XPU container:

```bash
docker run --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined ...
export RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1
ray start --head ...
```
