# XCCL / oneCCL Known Issues — Intel XPU Weight Sync

Hardware: 2× Intel Arc Pro B70 (Battlemage, PCIe, no Xe Link)
Stack: torch 2.12.0+xpu · oneCCL 2021.17 · vLLM 0.23.1rc1

These are the open issues that block or limit native XCCL weight sync
in OpenRLHF. Gloo (CPU-staged) is the current production default.

---

## Bug 1 — ProcessGroupXCCL.broadcast segfaults on torch 2.12

**Tracker:** intel/torch-xpu-ops #4238
**Status:** Fixed in torch 2.13.0+xpu (not yet validated with vLLM)

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

### Workaround
Use gloo (`--vllm.sync_backend gloo`). This is the OpenRLHF default for XPU.

### Unblock path
Rebuild vLLM-XPU against torch 2.13 — no compatible pairing exists yet.
See `backup/xpu-xccl-weight-sync` branch for the XCCL implementation.

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

## Summary table

| # | Issue | Tracker | Blocks XCCL E2E? | Workaround |
|---|---|---|---|---|
| 1 | ProcessGroupXCCL.broadcast segfault | torch-xpu-ops #4238 | **YES** (on torch 2.12) | Use gloo; fix: torch 2.13 |
| 2 | No PCIe P2P on B70 | compute-runtime #935/#942 | Performance ceiling only | Need shared PCIe switch |
| 3 | Stale IPC handle after vLLM realloc | oneCCL #212 | **YES** (hang) | `CCL_ZE_CACHE_*=0` |
| 4 | pidfd IPC deadlock | oneCCL #213 | **YES** (deadlock) | `CCL_ZE_IPC_EXCHANGE=sockets` |
| 5 | `ccl::reduction::avg` unsupported on GPU | MLSL-4181 | No (not used) | N/A |
| 6 | Wrong NIC selected in Docker | (untracked) | **YES** (hang in Docker) | `FI_PROVIDER=shm` |

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

Bug 1 (segfault) cannot be worked around via env vars — needs torch 2.13.
Bug 2 (no P2P) is a hardware limitation — needs a different box.

## What this means for PR3 (native XCCL)

PR3 can be submitted only after:
1. torch 2.13+xpu + vLLM ≥ 0.22 pairing is validated end-to-end
2. Bugs 3 and 4 confirmed still require env vars on that stack
3. Isolated broadcast test (`xccl_213_repro.py`) passes on the same vLLM build

See `backup/xpu-xccl-weight-sync` branch for the preserved implementation.
