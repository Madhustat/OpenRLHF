# Setting Up OpenRLHF on a New Intel XPU Machine — From Scratch

This document covers environment setup and installation only — no source code changes. For
the OpenRLHF source-code changes required to run on XPU, see the companion document
`PORTING_openrlhf_for_xpu.md`.

**Assumed hardware/OS:** Intel GPU with Level-Zero driver support, Ubuntu 24.04.x.

---

## Part 0 — Folder structure

Everything for this project lives under one dedicated folder, created directly inside your
base working directory (e.g. `~/madhu/`). Do not scatter the OpenRLHF clone, the Python
environment, and the vLLM source across different locations.

```bash
cd ~/madhu                     # or whatever your base folder is
mkdir openrlhf_exp
cd openrlhf_exp
```

Everything below — the OpenRLHF clone, the `venv-tf` Python environment, and the vLLM source
clone — is created as a direct child of `openrlhf_exp/`. By the end of this guide, the
structure looks like:

```
openrlhf_exp/
├── OpenRLHF/          (cloned in Part 1)
│   └── venv-tf/       (Python environment, created in Part 3, lives inside OpenRLHF/)
└── vllm-src/          (cloned in Part 8 — a SIBLING of OpenRLHF/, not nested inside it)
```

All commands below assume your shell's current directory is `openrlhf_exp/` unless a `cd` is
shown.

---

## Part 1 — Clone OpenRLHF

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
```

Everything from here on runs with `OpenRLHF/` as the current directory, unless stated
otherwise (Part 8, building vLLM, explicitly changes directory to the sibling `vllm-src/`).

Apply the source-code changes in `PORTING_openrlhf_for_xpu.md` to this clone before or after
the rest of this setup — order doesn't matter, but training will not run correctly until those
changes are applied.

---

## Part 2 — System-level (apt) packages

**Proxy note (corporate networks):** if any `apt`, `wget`, `curl`, or `pip` command below
times out or fails to resolve/connect, check `env | grep -i proxy` first. A `no_proxy`
environment variable that includes `intel.com`/`.intel.com` can cause tools to skip the proxy
specifically for Intel's own *public* package repos (`apt.repos.intel.com`), even though those
hosts are only reachable *through* the proxy, not directly. If so, unset it for the current
shell before continuing:
```bash
unset no_proxy NO_PROXY
```
`apt` and `wget`/`curl` do not always agree on how `no_proxy` is interpreted — if one tool
works and another doesn't with the same environment, this is the first thing to check.

### 2.1 — Required packages

```bash
sudo apt install -y libze-dev python3-dev gpg-agent wget
```

- **`libze-dev`** — provides the *unversioned* `libze_loader.so` symlink. Intel's oneCCL
  (backing PyTorch's `xccl` distributed backend) calls `dlopen("libze_loader.so")`; Ubuntu's
  Level-Zero runtime package only ships the versioned `libze_loader.so.1`. Without this,
  any real `torch.distributed` collective op over `xccl` fails with
  `oneCCL: ... EXCEPTION: ze_data was not initialized`.
- **`python3-dev`** — provides `Python.h`. DeepSpeed's `FusedAdam` (and other C++/SYCL
  extensions) compile as Python C-extension modules at first use and need this header.
  Without it: `fatal error: Python.h: No such file or directory`.
- **`gpg-agent`, `wget`** — needed for the next step (adding Intel's apt repository).

### 2.2 — Add Intel's oneAPI apt repository

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
```

Do not install the compiler package yet — Part 4 tells you which exact version to request,
based on a check you do in Part 3 first.

---

## Part 3 — Python environment and torch

```bash
python3 -m venv venv-tf
venv-tf/bin/pip install --upgrade pip
venv-tf/bin/pip install torch==2.12.0 torchvision --index-url https://download.pytorch.org/whl/xpu
venv-tf/bin/pip show intel-sycl-rt
```

Note the version number the last command prints — you need it for Part 4.

---

## Part 4 — Install the matching oneAPI DPC++/C++ compiler (`icpx`)

DeepSpeed's `FusedAdam` JIT-compiles a SYCL kernel on first use — this needs the compiler
binary itself, not just runtime libraries. The apt compiler version must match the
`intel-sycl-rt` version pip installed with torch (Part 3); a mismatch produces build errors
like `unknown type name '__DPCPP_SYCL_EXTERNAL_LIBC'`.

```bash
apt-cache madison intel-oneapi-compiler-dpcpp-cpp-2025.3
```
Find the package version matching the `intel-sycl-rt` number from Part 3, then install it:
```bash
sudo apt install -y intel-oneapi-compiler-dpcpp-cpp-2025.3=<matching-version>
```

---

## Part 5 — Add the compiler and venv to `PATH`

```bash
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$(pwd)/venv-tf/bin:$PATH"
```

**Do NOT** source the compiler's `env/vars.sh` script instead of this line. Sourcing that
script prepends the compiler's own `lib/` directories to `LD_LIBRARY_PATH`, which shadows the
correctly-versioned Level-Zero/SYCL runtime libraries torch depends on, breaking XPU device
detection entirely. Adding only `bin/` to `PATH` is sufficient.

**Verify both binaries are now reachable:**
```bash
which icpx
which ninja   # will fail until Part 6 installs the ninja package — re-check after that step
```
`icpx` should point inside `/opt/intel/oneapi/compiler/2025.3/bin/`.

**Run this same `export` command every time you open a new terminal for this project** — it
does not persist across sessions on its own.

---

## Part 6 — OpenRLHF's Python dependencies

### 6.1 — Install requirements, excluding `flash-attn`

```bash
grep -v "^flash-attn" requirements.txt > /tmp/requirements_no_flashattn.txt
venv-tf/bin/pip install -r /tmp/requirements_no_flashattn.txt
```
`flash-attn` cannot be installed on Intel XPU — it compiles CUDA kernels via `nvcc`, and there
is no CUDA toolchain here. `PORTING_openrlhf_for_xpu.md` covers the source change that routes
around this dependency.

### 6.2 — Version bumps and extras

```bash
venv-tf/bin/pip install "transformers>=5.13.0" "kernels>=0.15.2,<0.16.0" mpi4py ninja
```
- `transformers>=5.13.0` — required for the flash-attention replacement used by the porting
  guide's fix (`_pad_input`/`_unpad_input`/`_index_first_axis` live in
  `transformers/modeling_flash_attention_utils.py` from this version on).
- `kernels` — backs `attn_implementation="flash_attention_2"`'s auto-redirect to a
  pre-compiled Intel SYCL kernel. **Do not use `--ds.packing_samples`** together with this —
  that flag requires `kernels<0.16.0` while the flash-attention path needs `kernels==0.16.0`;
  the two are mutually exclusive as installed here.
- `mpi4py` — needed if launching via plain `python -m` instead of the `deepspeed`/`torchrun`
  launcher; DeepSpeed's `init_distributed()` falls back to MPI-based rank discovery when
  standard launcher env vars aren't present.
- `ninja` — installs the `ninja` executable that DeepSpeed's JIT-compile path calls directly.

**Verify `ninja` is now reachable** (re-run the check from Part 5):
```bash
which ninja
```
Should now print a path inside `venv-tf/bin/`. If it doesn't, this `PATH` export from Part 5
hasn't taken effect in your current shell — re-run it.

### 6.3 — Install OpenRLHF itself, editable

```bash
venv-tf/bin/pip install -e . --no-deps
```

---

## Part 7 — Verify torch sees the XPU correctly

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('torch.xpu.is_available():', torch.xpu.is_available())
print('torch.accelerator.current_accelerator():', torch.accelerator.current_accelerator())
"
```
Expected: `2.12.0+xpu`, `True`, `xpu`. If `torch.accelerator.current_accelerator()` returns
`None` or errors, stop here — nothing past this point will work. Re-check the `PATH` export in
Part 5 first (the `env/vars.sh` sourcing mistake is the most common cause).

```bash
xpu-smi discovery -d 0
```
Expect `Device State: normal`.

---

## Part 8 — Build vLLM for XPU

vLLM is built from source, as a **sibling** of `OpenRLHF/`, not nested inside it — cloning it
inside `OpenRLHF/` breaks `import vllm` silently (Python's namespace-package resolution gets
confused between the real installed `vllm` package and a same-named directory reachable from
`OpenRLHF/`'s own working directory).

```bash
cd ..                              # back to openrlhf_exp/
git clone https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src
../OpenRLHF/venv-tf/bin/pip install -v -r requirements/xpu.txt
```
This pins `torch==2.12.0` (matches Part 3) and installs `vllm_xpu_kernels` — a pre-built wheel
with Intel's SYCL kernels — automatically.

**Fix the `triton`/`triton-xpu` conflict** (plain `triton` gets pulled in transitively and
overwrites files `triton-xpu` needs, breaking Intel's backend):
```bash
../OpenRLHF/venv-tf/bin/pip uninstall -y triton triton-xpu
../OpenRLHF/venv-tf/bin/pip install triton-xpu==3.7.1 --extra-index-url https://download.pytorch.org/whl/xpu --no-deps
```
Verify: `../OpenRLHF/venv-tf/bin/python -c "from triton.backends import backends; print(backends.keys())"` must show `'intel'`.

**Build and install (editable):**
```bash
VLLM_TARGET_DEVICE=xpu ../OpenRLHF/venv-tf/bin/pip install --no-build-isolation -e . -v
```
**Re-run the `triton`/`triton-xpu` fix above again after this build finishes — do not skip
this, it is not a rare edge case.** Confirmed to recur every time on two separate machines: this
`pip install -e .` step pulls in `xgrammar` as a vLLM dependency, and `xgrammar` itself depends
on plain `triton`, which pip installs automatically and which overwrites `triton-xpu`'s files
in the same `site-packages/triton/` path. Re-run all three commands (uninstall, reinstall
`triton-xpu`, verify `'intel'` is in the backends list) immediately after this build step
finishes, every time, on every machine — treat it as part of this step, not an optional
troubleshooting fallback.

**Verify real vLLM inference on XPU:**
```bash
../OpenRLHF/venv-tf/bin/python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', dtype='bfloat16', gpu_memory_utilization=0.5, enforce_eager=True)
outputs = llm.generate(['The capital of France is'], SamplingParams(max_tokens=10, temperature=0))
print(outputs[0].outputs[0].text)
"
```
Expected: coherent generated text (e.g. `Paris. It is the largest city in Europe and`). Check
the logs for `backend=xccl` and `Using Flash Attention backend` / `FlashAttention version 2` —
confirms vLLM's XPU-native path is engaged, not a silent fallback.

**Note on `torchcodec`:** the XPU requirements pull in `torchcodec` with a `+cpu` build suffix
(not `+xpu`) — this is expected, not a broken install. Confirm with
`../OpenRLHF/venv-tf/bin/pip show torchcodec` if you want to double check; a `+cpu` version
here does not indicate the rest of the stack (`torch`, `torchvision`, `torchaudio`) is wrong —
those should still show `+xpu`.

```bash
cd ../OpenRLHF
```

---

## Part 9 — Ray: start with GPU count declared

Ray's automatic GPU detection is NVIDIA/CUDA-specific and does not recognize Intel XPU devices
— a bare `ray.init()` registers zero GPU resources. If your training config uses
`--train.colocate_all` (or any placement group requiring a `{"GPU": 1}` bundle), the placement
group can never become ready and the process hangs indefinitely inside `ray.get(pg.ready())`.

**Before running any `train_ppo_ray.py` command that uses placement groups, start Ray manually
with the GPU count declared:**
```bash
ray stop --force && rm -rf /tmp/ray
NUM_GPUS=$(python -c "import torch; print(torch.xpu.device_count())")
ONEAPI_DEVICE_SELECTOR=level_zero:0 ray start --head --num-gpus="$NUM_GPUS"
```
`train_ppo_ray.py` checks `ray.is_initialized()` and attaches to this cluster instead of
creating its own. Verify:
```bash
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```
Must show a `GPU` key with the correct count.

**After editing any file Ray actors import** (anything under `openrlhf/trainer/ray/`), restart
the cluster (`ray stop --force && rm -rf /tmp/ray`, then the commands above again) — Ray's
worker processes cache imported code and will silently keep running old versions otherwise.

---

## Part 10 — Run the unit test suite

If this is a new terminal session, activate the environment first:
```bash
cd OpenRLHF
source venv-tf/bin/activate
export PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"
```

`pytest` is not in OpenRLHF's `requirements.txt`, so it must be installed separately:
```bash
pip install pytest
```

Then run the suite:
```bash
python -m pytest tests/ -v
```
All tests should pass before attempting real training.

---

## What to bring over from the reference environment

If copying files from an existing working setup rather than following this guide purely from
the public repo:

- **`PORTING_openrlhf_for_xpu.md`** — required. Contains the exact source-code changes (with
  before/after code) to apply to this fresh clone. Nothing here works on XPU without these.
- **`tests/test_loss_aggregation_generic.py`** and **`tests/test_distributed_backend_generic.py`**
  — new files, not derived from a diff against upstream. Copy these directly into `tests/`.
- **`tests/test_ray_env_vars.py`** — modified from upstream (2 tests added). Either copy this
  file directly, or apply the corresponding change described in
  `PORTING_openrlhf_for_xpu.md`.
- No separate `requirements.txt` is needed — the repo's own `requirements.txt` (Part 6.1) is
  used as-is, with `flash-attn` filtered out at install time.

## Summary checklist

- [ ] (Part 0) Create `openrlhf_exp/` as the single base folder for everything
- [ ] (Part 1) Clone OpenRLHF into it
- [ ] (Part 2) `apt install libze-dev python3-dev gpg-agent wget`; add Intel's oneAPI apt repo
- [ ] (Part 3) Create `venv-tf`, install `torch==2.12.0+xpu`, note `intel-sycl-rt` version
- [ ] (Part 4) Install the matching oneAPI compiler version
- [ ] (Part 5) Export `PATH` (compiler `bin/` + venv `bin/`) — every new terminal session
- [ ] (Part 6) Install OpenRLHF's requirements (minus `flash-attn`), plus `transformers`,
      `kernels`, `mpi4py`, `ninja`; install OpenRLHF itself editable
- [ ] (Part 7) Verify `torch.xpu.is_available()` / `torch.accelerator.current_accelerator()`
- [ ] (Part 8) Clone vLLM as a sibling of `OpenRLHF/`; install requirements; fix
      `triton`/`triton-xpu`; build; re-fix `triton`/`triton-xpu`; verify generation
- [ ] (Part 9) Start Ray manually with `--num-gpus=N` before any placement-group-based run
- [ ] (Part 10) `pip install pytest` (not in requirements.txt), then run `pytest tests/`,
      confirm all pass
- [ ] Apply `PORTING_openrlhf_for_xpu.md`'s source changes and copy over the test files listed
      above before attempting real training
