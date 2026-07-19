# Setting Up OpenRLHF on an NVIDIA GPU Machine — Generic Procedure

This follows OpenRLHF's own official installation procedure (as documented in the project's
`README.md`), for NVIDIA/CUDA hardware. Unlike the Intel XPU setup, no separate vendor compiler
toolchain (equivalent to Intel's oneAPI `icpx`) needs to be installed manually — NVIDIA's CUDA
toolkit and PyTorch's own CUDA wheels handle the equivalent role, and `flash-attn` installs
directly via pip (it does not need to be routed around, unlike on XPU).

**No Intel-specific setup applies here at all** — skip `libze-dev`, oneAPI compiler
installation, and any `PATH` exports for `icpx`. None of that is relevant on NVIDIA hardware.

---

## Part 0 — Folder structure

Same convention as the XPU setup: one dedicated project folder, everything as a direct child
of it.

```bash
cd ~/madhu                     # or whatever your base folder is
mkdir openrlhf_exp
cd openrlhf_exp
```

---

## Part 1 — Recommended: Docker-based install

OpenRLHF's own documented, recommended approach uses NVIDIA's official PyTorch container,
which already has CUDA, PyTorch, and the necessary build toolchain preinstalled.

**Before running `docker run`, `cd` into the exact folder you want mounted.** The `-v
$PWD:/openrlhf` flag mounts whatever your current directory is *at the moment you run the
command* — if you run it from the wrong directory, the container's `/openrlhf` will point
somewhere unexpected. If you intend to work on a source checkout (Part 2 below), run this
from inside that checkout, not its parent.

```bash
cd ~/madhu/openrlhf_exp/OpenRLHF   # or wherever your checkout actually is
```

**Corporate/proxy networks:** if this host requires a proxy for internet access (check
`cat /etc/environment | grep -i proxy` — proxy variables set there are NOT automatically
inherited by Docker containers and must be passed in explicitly), include them in the
`docker run` command:
```bash
docker run --runtime=nvidia -it --name openrlhf-nvidia --shm-size="10g" --cap-add=SYS_ADMIN \
  -e http_proxy="<your proxy URL>" -e https_proxy="<your proxy URL>" \
  -e HTTP_PROXY="<your proxy URL>" -e HTTPS_PROXY="<your proxy URL>" \
  -e no_proxy="<your no_proxy value>" \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:26.03-py3 bash
```
If no proxy is needed, drop the `-e` lines. **Use `--name openrlhf-nvidia` (or any name) instead
of `--rm`** — `--rm` destroys the container and everything installed inside it the moment you
exit, forcing a full reinstall next time. A named container can be stopped and resumed later
with everything still installed intact (see the companion file
`DOCKER_start_or_resume_nvidia.md` for the exact resume commands).

This drops you into a shell **inside the container**, as `root` (e.g. `root@<container-id>:/workspace#`)
— NVIDIA's PyTorch container images run as `root` by default. This is expected and unrelated
to your host machine's user account; you are now in an isolated environment, separate from the
host filesystem except for whatever you mounted with `-v`.

Because you're already `root`, there is no `sudo` command inside this container (you'll get
`sudo: command not found` if you try it) — just run commands directly, without `sudo`. Confirm
network access works before installing anything:
```bash
python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"
```
Should print `200`. If it raises `OSError: [Errno 101] Network is unreachable`, the proxy
wasn't passed in correctly — exit and re-run `docker run` with the `-e` proxy flags above.

Uninstall the packages that conflict with OpenRLHF's own dependency versions:
```bash
pip uninstall xgboost transformer_engine flash_attn pynvml -y
```

**`flash-attn` needs to be installed separately first, without build isolation** — its build
script imports `torch` directly, but pip's isolated build environment doesn't include
already-installed packages by default, so a plain `pip install openrlhf[vllm]` fails partway
through with `ModuleNotFoundError: No module named 'torch'` while trying to build `flash-attn`.
Install it explicitly first so it can see the container's existing torch:
```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Then install OpenRLHF itself (choose one, based on whether you need vLLM rollout support):
```bash
pip install openrlhf                    # Basic
pip install openrlhf[vllm]              # + vLLM 0.22.1 (recommended)
pip install openrlhf[vllm_latest]       # + Latest vLLM
pip install openrlhf[vllm,ring,liger]   # + All optimizations
```

This is sufficient for a working installation if you don't need to modify OpenRLHF's own
source code. If you do (e.g. to apply local changes, or to match exactly what's being tested
on the XPU side), use Part 2 instead.

---

## Part 2 — Install from source (for local source changes)

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
```

Create a Python virtual environment and install torch with CUDA support:
```bash
python3 -m venv venv-tf
venv-tf/bin/pip install --upgrade pip
venv-tf/bin/pip install torch torchvision
```
(No `--index-url` override is needed here, unlike XPU — PyPI's default `torch` wheel already
targets CUDA on Linux.)

Install OpenRLHF's requirements as-is — no filtering needed. `flash-attn==2.8.3` (pinned in
`requirements.txt`) installs directly on NVIDIA hardware, since it compiles against `nvcc`
from the CUDA toolkit, or picks up a matching prebuilt wheel:
```bash
venv-tf/bin/pip install -r requirements.txt
```

Install OpenRLHF itself, editable:
```bash
venv-tf/bin/pip install -e . --no-deps
```

Activate the environment for the rest of this guide:
```bash
source venv-tf/bin/activate
```

---

## Part 3 — Verify torch sees the GPU correctly

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.accelerator.current_accelerator():', torch.accelerator.current_accelerator())
"
```
Expected: a `+cuXXX`-suffixed torch version (e.g. `2.x.x+cu121`), `True`, `cuda`.

```bash
nvidia-smi
```
Confirms the GPU(s) are visible and healthy at the driver level.

---

## Part 4 — vLLM (if not already included via `openrlhf[vllm]` in Part 1)

If you installed via Part 2 (from source) and need vLLM rollout support, install it directly —
no custom build from source is required on NVIDIA, unlike XPU:
```bash
pip install vllm
```
OpenRLHF's README recommends **vLLM 0.22.1+** for best performance. See the project's
[Dockerfiles](https://github.com/OpenRLHF/OpenRLHF/tree/main/dockerfile) and
[Nvidia-Docker Install Script](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/nvidia_docker_install.sh)
for additional reference configurations.

Verify:
```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', dtype='bfloat16', gpu_memory_utilization=0.5, enforce_eager=True)
outputs = llm.generate(['The capital of France is'], SamplingParams(max_tokens=10, temperature=0))
print(outputs[0].outputs[0].text)
"
```

---

## Part 5 — Ray

Unlike Intel XPU, Ray's GPU auto-detection works natively for NVIDIA hardware — a bare
`ray.init()` correctly registers the visible CUDA devices. No manual `--num-gpus` override or
`ONEAPI_DEVICE_SELECTOR`-style workaround is needed.

If running into GPU device index issues with DeepSpeed under Ray, OpenRLHF's own README notes
this environment variable as a fix:
```bash
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

For Ray-based multi-node/auto-deploy setups, OpenRLHF's README also documents letting Ray
install the package itself on worker nodes:
```
--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'
```

---

## Part 6 — Run the unit test suite

`pytest` is not in OpenRLHF's `requirements.txt` — install it separately:
```bash
pip install pytest
```

Make sure you're in the OpenRLHF checkout directory itself before running pytest — inside the
container this is wherever your mount landed (e.g. `/openrlhf`, or a nested path like
`/openrlhf/<subfolders>/OpenRLHF` if you ran `docker run` from a parent directory rather than
the checkout itself; use `find / -maxdepth 5 -iname "test_ray_env_vars.py" 2>/dev/null` inside
the container if you're unsure where it landed). Then run:
```bash
python -m pytest tests/ -v
```

**Bring over these two additional files** (not part of upstream OpenRLHF) if you want the same
device-generic test coverage validated on the XPU side:
- `tests/test_loss_aggregation_generic.py`
- `tests/test_distributed_backend_generic.py`

Both are hardware-agnostic by design — they parametrize over `["cpu"] + [whatever
torch.accelerator detects]`, so on this NVIDIA machine they will automatically run their `cpu`
case plus a `cuda` case (in place of the `xpu` case seen on Intel hardware), with no code
changes needed. This is the entire point of writing them against `torch.accelerator` rather
than a hardcoded device string.

The `test_ray_env_vars.py` modifications made on the XPU side (2 additional tests for
`CCL_LOG_LEVEL`) are not necessary here — `CCL_LOG_LEVEL` is Intel oneCCL/XCCL-specific and has
no meaning on NVIDIA/NCCL. Upstream `test_ray_env_vars.py`, unmodified, is sufficient.

---

## Summary checklist

- [ ] (Part 0) Create `openrlhf_exp/` as the base folder
- [ ] (Part 1) Either: pull the NVIDIA PyTorch Docker image and `pip install openrlhf[vllm]`,
      **or** (Part 2) clone from source and set up a venv
- [ ] (Part 3) Verify `torch.cuda.is_available()` and `nvidia-smi` both report correctly
- [ ] (Part 4) Confirm vLLM generation works, if using vLLM rollout
- [ ] (Part 5) No manual Ray GPU configuration needed (unlike XPU) — confirm
      `ray.cluster_resources()` shows a `GPU` key automatically
- [ ] (Part 6) `pip install pytest`, copy over the 2 generic test files, run `pytest tests/`
