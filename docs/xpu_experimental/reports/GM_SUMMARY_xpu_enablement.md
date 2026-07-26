# Enabling Intel XPU for OpenRLHF

**OpenRLHF — a leading open-source RLHF training framework.**

Bringing it to Intel XPU lets our accelerators run mainstream reinforcement-learning
post-training (the technique behind aligning modern LLMs), on the same code NVIDIA users run.

---

## What we achieved

- **OpenRLHF now runs on Intel XPU.** All major RLHF training workflows were smoke-tested and
  run end-to-end on XPU.
- **NVIDIA is unaffected.** The changes are device-agnostic — the same code adapts to the
  hardware automatically, so NVIDIA users see no change and Intel gains support.
- **3 PRs submitted upstream**, each backed by automated tests.
- **Unit tests: 100% pass** on the XPU-runnable suite.

---

## Unit tests at a glance

| | Count |
|---|---|
| Total unit tests | 21 |
| Runnable on XPU | 12 |
| **Passing on XPU** | **12 (100%)** |
| Pure-software tests passing | 9 / 9 |
| Failures / blocked | 0 |
| Newly added for XPU | 4 |

*In short: everything that can run on XPU passes, nothing is failing, and we added new tests to
lock in the XPU behavior.*

---

## The 3 submitted PRs

| PR | In plain terms |
|---|---|
| **oneCCL logging** | Makes Intel's communication library log correctly alongside NVIDIA's. |
| **Loss-aggregation tests** | Existing tests now actually exercise the accelerator, not just CPU. |
| **Distributed-backend test** | Proves real multi-process communication works on each vendor's hardware. |

All three are written to work on both NVIDIA and Intel from one code path.

---

## Still in progress (WIP)

Additional, larger enablement is implemented and passing on a single XPU, and is being hardened
before upstreaming:

- Device-agnostic API migration across the framework (the bulk of the enablement).
- vLLM integration on XPU — device assignment and model weight synchronization.
- Making a CUDA-only attention dependency optional so XPU can run without it.

These work today on one XPU; what remains is **validation on larger hardware**.

---

## Next steps

1. Validate on **multi-XPU** and **NVIDIA** hardware (the pieces we can't test on a single-GPU box).
2. Finish hardening and **upstream the remaining enablement PRs**.

---

## Ask

Access to **multi-XPU and NVIDIA hardware** to complete final validation before upstreaming.
