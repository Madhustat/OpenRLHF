# XCCL / oneCCL Known Issues — Intel XPU Weight Sync

Hardware: 2× Intel Arc Pro B70 (Battlemage, PCIe, no Xe Link)
Stack (original): torch 2.12.0+xpu · oneCCL 2021.17 · vLLM 0.23.1rc1
Stack (2026-09-06 update below): torch 2.13.0+xpu · vLLM 0.27.2.dev0
(built natively from the v0.27.1 tag) · native venv, no Docker

**UPDATE (2026-09-06):** Bug 1 below is now E2E-validated fixed on the newer
stack — XCCL is no longer gated to gloo-only. `resolve_vllm_sync_backend()`'s
auto-detect now picks xccl by default on >=2 physical XPUs with torch>=2.13
(see the `xpu/xccl-auto-detect` branch). Gloo remains the default below that
version or on a single XPU, and remains available as an explicit override
(`--vllm.sync_backend gloo`). See Bug 7 (new, below) for a caveat found while
validating this: ZeRO-3 specifically hit a real GPU engine reset in this exact
combination that ZeRO-2 did not.

These are the open issues that block or limit native XCCL weight sync
in OpenRLHF. Gloo (CPU-staged) is the current production default.

---

## Bug 1 — ProcessGroupXCCL.broadcast segfaults on torch 2.12

**Tracker:** intel/torch-xpu-ops #4238
**Status:** Fixed in torch 2.13.0+xpu — **E2E-validated 2026-09-06** (see below)

### What happens
On torch 2.12.0+xpu, calling `ProcessGroupXCCL.broadcast` crashes with:
```
ProcessGroupXCCL.cpp:1916  device ... unknown
terminate called (SIGABRT)
```
The broadcast segfaults on the first weight-sync step. Training cannot
proceed with XCCL on this torch version.

### Root cause
The XCCL ProcessGroup does not have a definite rank→device mapping at the
point the broadcast fires. Level-Zero logs the device as "unknown" and the
collective hangs or crashes.

### What we verified
- Isolated 2-rank broadcast: **FAILS** on torch 2.12, **PASSES** on torch 2.13.dev
- E2E GRPO training with XCCL: **0 steps** on torch 2.12
- E2E GRPO training with gloo: **EXIT 0, rewards move**
- **(2026-09-06, native torch 2.13.0+xpu + vLLM 0.27.2.dev0, no Docker) E2E
  REINFORCE with `--vllm.sync_backend xccl`, disjoint actor+vLLM on separate
  XPUs: PASS 10/10 steps.** Repeated under `--train.colocate_all` (actor+vLLM
  sharing GPUs via sleep/wake): PASS 3/3 (REINFORCE) and PASS 4/4 (PPO+critic,
  ZeRO-2). None of Bug 3/4/6's env-var workarounds below (`CCL_ZE_IPC_EXCHANGE`,
  `FI_PROVIDER`, `CCL_ZE_CACHE_*`) were set for any of these runs and none of
  those failure modes reproduced — not confirmed fixed, just not hit; this
  native stack may differ enough from the original Docker+torch-2.12 stack
  those bugs were filed against that they no longer apply, or these runs
  simply didn't run long/hard enough to trigger them.

### Workaround (still valid below torch 2.13, or on a single XPU)
Use gloo (`--vllm.sync_backend gloo`), or let auto-detect choose it for you.

### Unblock path — DONE
~~Rebuild vLLM-XPU against torch 2.13 — no compatible pairing exists yet.~~
Done: built natively from vLLM's `v0.27.1` tag against torch 2.13.0+xpu
(`use_existing_torch.py` + `requirements/xpu.txt`, which itself now pins
`torch==2.13.0`). See `backup/xpu-xccl-weight-sync` branch for the original
XCCL implementation this was built on.

---

## Bug 2 — PCIe P2P unavailable on B70 across separate root ports

**Tracker:** intel/compute-runtime #935 / #942
**Status:** Open — hardware limitation on B70 without a shared PCIe switch

### What happens
True device-to-device PCIe P2P is unavailable between the two B70 cards
on this box. ACS-off does not help because the cards sit behind different
root ports. The kernel refuses P2PDMA without a shared upstream switch.

### Impact on XCCL
Even when XCCL works (torch 2.13), all transfers are host-staged via CPU
memory. There is no direct XPU→XPU DMA path. XCCL still beats gloo
(~1.8× allreduce speed per Intel benchmarks) because it avoids Python
overhead, but it is not the "zero CPU hop" ideal.

### Unblock path
Requires a shared PCIe switch upstream of both B70 cards, plus
`iommu=pt` kernel flag. Not achievable on the current test box.

---

## Bug 3 — Stale IPC handle after vLLM buffer reallocation

**Tracker:** uxlfoundation/oneCCL #212
**Status:** Open

### What happens
oneCCL caches opened Level-Zero IPC handles. When vLLM reallocates its
KV cache or weight buffers, the cached handle becomes stale. The next
XCCL collective dereferences the stale handle → GPU page fault → infinite
hang.

### When it triggers
Happens when XCCL is used alongside vLLM in the same training run (the
exact OpenRLHF use case). Pure XCCL benchmarks without vLLM do not hit it.

### Workaround
```bash
export CCL_ZE_CACHE_OPEN_IPC_HANDLES=0
export CCL_ZE_CACHE_GET_IPC_HANDLES=0
```
Disables IPC handle caching. Adds small overhead per collective but
prevents the hang. Already in `xccl_env.sh`.

---

## Bug 4 — pidfd IPC exchange deadlocks in default mode

**Tracker:** uxlfoundation/oneCCL #213
**Status:** Open

### What happens
oneCCL's default IPC exchange mechanism (`pidfd`) deadlocks in some
configurations (observed in Docker and on the B70 test box). In Docker,
`CCL_ZE_IPC_EXCHANGE=sockets` is silently ignored in some versions,
making the deadlock non-obvious.

### Workaround
```bash
export CCL_ZE_IPC_EXCHANGE=sockets
```
Switches from pidfd to UNIX socket-based IPC exchange, which is stable
on this hardware. Already in `xccl_env.sh`.

---

## Bug 5 — ccl::reduction::avg not supported on GPU scheduler path

**Tracker:** Intel internal MLSL-4181
**Status:** Open

### What happens
Calling `ccl::reduction::avg` in a collective on GPU throws:
```
oneCCL: coll_param.cpp:458 validate:
EXCEPTION: average operation is not supported for the scheduler path
```
Fails identically with MPI and OFI transport layers.

### Impact on OpenRLHF
**Does not affect us.** OpenRLHF uses `broadcast` for weight sync and
`allreduce` with `sum` (via DeepSpeed) for gradient sync. `avg` is never
called. Listed here for completeness.

---

## Bug 6 — oneCCL selects wrong NIC in Docker (br-* bridge interface)

**Not a tracked issue — observed behaviour**
**Status:** Workaround available

### What happens
Inside Docker, oneCCL auto-selects the Docker bridge NIC (`br-xxxxxxxx`)
for ATL transport instead of the intended shared-memory path. The bridge
interface is not a valid path for GPU collectives and hangs.

### Workaround
```bash
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
```
Forces the shared-memory OFI provider. Already in `xccl_env.sh`.

---

## Bug 7 — GPU compute-engine (ccs) timeout/reset under ZeRO-3+colocate_all+xccl+critic

**Tracker:** none yet — not filed upstream
**Status:** Open, intermittent, not root-caused

### What happens
PPO+critic (GAE), `--train.colocate_all --vllm.enable_sleep --ds.enable_sleep`,
`--vllm.sync_backend xccl`, **ZeRO stage 3**: 3 global steps complete correctly,
then step 4's weight broadcast (inside DeepSpeed's ZeRO-3 parameter all-gather,
`partition_parameters.py:_allgather_params_sequential ->
get_accelerator().synchronize()`) raises:
```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```
Confirmed via `dmesg` at the exact crash timestamp (not inferred from the
Python traceback alone):
```
xe 0000:54:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs ... Timedout job ... in ray::IDLE [<pid>]
xe 0000:18:00.0: [drm] Tile0: GT0: Engine reset: engine_class=ccs ...   (1 second later, other card)
```
Both cards self-recover to `Device State: normal` (`xpu-smi discovery`)
immediately after — not a dead card, an intermittent timeout under this
specific load pattern. The exact same topology at **ZeRO-2 instead of ZeRO-3
ran clean (4/4 steps)** — the leading (unconfirmed) theory is ZeRO-3's extra
all-gather/synchronize traffic during the weight-gather-before-broadcast step
pushes something (thermal, power, driver scheduling) over a threshold that
ZeRO-2 doesn't reach. This same PCI address (`54:00.0`) had two earlier,
separate coredump events on 2026-09-02 (both auto-recovered) — not a one-off.

### Impact on OpenRLHF
Only observed in the ZeRO-3 + `colocate_all` + xccl + critic combination.
ZeRO-2 with the same topology, and xccl without `colocate_all` (disjoint
actor/vLLM GPUs), have both been clean across multiple runs.

### Workaround
Use `--vllm.sync_backend gloo` explicitly for ZeRO-3 + critic + colocate_all
until this is root-caused or reproduced enough times to characterize.

### Unblock path
Reproduce multiple times to determine if it's deterministic-under-load or
genuinely rare/flaky; if deterministic, bisect against ZeRO-2 to isolate
which specific all-gather call triggers it; consider filing against
intel/torch-xpu-ops with the dmesg evidence once characterized.

---

## Summary table

| # | Issue | Tracker | Blocks XCCL E2E? | Workaround |
|---|---|---|---|---|
| 1 | ProcessGroupXCCL.broadcast segfault | torch-xpu-ops #4238 | **NO** (fixed + E2E-validated on torch 2.13, 2026-09-06) | Use gloo below torch 2.13 |
| 2 | No PCIe P2P on B70 | compute-runtime #935/#942 | Performance ceiling only | Need shared PCIe switch |
| 3 | Stale IPC handle after vLLM realloc | oneCCL #212 | Not reproduced on the torch-2.13 native stack (unconfirmed fixed) | `CCL_ZE_CACHE_*=0` if it recurs |
| 4 | pidfd IPC deadlock | oneCCL #213 | Not reproduced on the torch-2.13 native stack (unconfirmed fixed) | `CCL_ZE_IPC_EXCHANGE=sockets` if it recurs |
| 5 | `ccl::reduction::avg` unsupported on GPU | MLSL-4181 | No (not used) | N/A |
| 6 | Wrong NIC selected in Docker | (untracked) | N/A on native (no Docker bridge) | `FI_PROVIDER=shm` in Docker |
| 7 | GPU ccs engine timeout/reset, ZeRO-3+colocate_all+xccl+critic | none yet | **YES**, that specific combo only | Use gloo for that combo |

## Current env fix (xccl_env.sh)

All workarounds for bugs 3, 4, 6 are captured in one file:

```bash
export CCL_ZE_IPC_EXCHANGE=sockets      # bug 4: avoid pidfd deadlock
export FI_PROVIDER=shm                  # bug 6: avoid Docker bridge NIC
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES=0  # bug 3: avoid stale IPC handle
export CCL_ZE_CACHE_GET_IPC_HANDLES=0
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
```

Bug 1 (segfault) — resolved, see above; no longer needs these env vars on the
native torch-2.13 stack (not confirmed why — possibly bugs 3/4/6 themselves
are also resolved on this stack, possibly just not triggered yet).
Bug 2 (no P2P) is a hardware limitation — needs a different box.
Bug 7 (new) — GPU engine reset specific to ZeRO-3+colocate_all+xccl+critic;
use gloo for that specific combination until characterized further.

## What this means for PR3 (native XCCL) — UPDATE 2026-09-06

The 3 original unblock conditions are now met:
1. ~~torch 2.13+xpu + vLLM ≥ 0.22 pairing validated end-to-end~~ — **DONE**,
   native (no Docker): torch 2.13.0+xpu + vLLM 0.27.2.dev0 (built from the
   v0.27.1 tag). REINFORCE and PPO+critic both pass with real weight-sync.
2. Bugs 3/4 not reproduced without their env-var workarounds on this stack
   (see Bug 1 update) — not the same as "confirmed fixed," but not blocking.
3. ~~Isolated broadcast test passes~~ — **DONE**, and taken further: real E2E
   training passes too, not just the isolated broadcast.

`resolve_vllm_sync_backend()`'s auto-detect now picks xccl by default on
>=2 physical XPUs + torch>=2.13 (branch `xpu/xccl-auto-detect`, on top of
`openrlfh_exp_multi`). **New caveat before calling this fully done: Bug 7**
(ZeRO-3+colocate_all+xccl+critic GPU engine reset) needs to be understood —
auto-selection does not currently distinguish ZeRO stage, so that specific
combination should use `--vllm.sync_backend gloo` explicitly until Bug 7 is
characterized or fixed.

See `backup/xpu-xccl-weight-sync` branch for the preserved original
implementation this was built on.
