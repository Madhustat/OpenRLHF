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

For a shared machine or the 2-GPU E2E suite, expose only two known-free GPUs instead of every
GPU on the host. Check ownership first with `nvidia-smi`, then replace `--runtime=nvidia` with
an explicit GPU request such as this (example selects host GPUs 0 and 1):
```bash
docker run --gpus '"device=0,1"' -it --name openrlhf-nvidia-e2e --ipc=host --shm-size=16g \
  -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" \
  -e HTTP_PROXY="$HTTP_PROXY" -e HTTPS_PROXY="$HTTPS_PROXY" \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:26.03-py3 bash
```
The container sees its selected devices as CUDA devices 0 and 1, so use
`CUDA_VISIBLE_DEVICES=0,1` inside the container. Do not select a host GPU already used by
another workload.

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

For the direct DeepSpeed commands used by the SFT, reward-model, and DPO E2E tests, also install
the MPI Python binding. Without it, those tests stop before training with
`ModuleNotFoundError: No module named 'mpi4py'`:
```bash
pip install mpi4py
python -c "from mpi4py import MPI; print('mpi4py OK; MPI', MPI.Get_version())"
```
If pip must compile the binding, point it at the MPI compiler supplied by the container, for
example `MPICC=/usr/local/mpi/bin/mpicc pip install mpi4py`. This requirement is for the
supervised DeepSpeed launch path, not a CUDA/NCCL failure.

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

For the multi-GPU suite, verify the container sees at least two selected devices before starting:
```bash
python -c "import torch; print('cuda:', torch.version.cuda, 'device_count:', torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() >= 2"
```

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

## Part 7 — CUDA multi-GPU E2E suite

The branch's `tests/test_e2e_suite_multigpu.sh` auto-detects CUDA, sets
`CUDA_VISIBLE_DEVICES=0,1` when it is not already set, chooses the `nccl` weight-sync backend,
and exports `DS_SKIP_CUDA_CHECK=1` for the DeepSpeed `adam_offload` JIT. No XPU variables should
be set for this path.

From the repository root, use the two devices exposed to the container:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

python -c "
import torch, ray, deepspeed, transformers, vllm
from mpi4py import MPI
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())
print('ray', ray.__version__, 'deepspeed', deepspeed.__version__)
print('transformers', transformers.__version__, 'vllm', vllm.__version__)
print('MPI', MPI.Get_version())
"
```

The multi-GPU suite has one VLM test. Download its required local model before the full run;
otherwise `vlm_grpo` fails before training because the configured local path is absent:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
  'Qwen/Qwen2-VL-2B-Instruct',
  local_dir='/tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct',
)
"
```
This snapshot is about 4.2 GB in the validated CUDA environment. Ensure the container filesystem
has sufficient free space for it, the Hugging Face cache, Ray temporary files, and checkpoints.

Smoke-test the two baseline configurations first:
```bash
bash tests/test_e2e_suite_multigpu.sh grpo_zero
```
The banner must include:
```text
Device: cuda | Sync backend: nccl
```
Both `grpo_zero2_nocolo` and `grpo_zero3_nocolo` should reach 10 steps. Then start the full run:
```bash
nohup bash tests/test_e2e_suite_multigpu.sh > tests/multigpu_run.log 2>&1 &
echo "PID=$!"
```

Monitor it and inspect the latest summary with:
```bash
tail -f tests/multigpu_run.log
RDIR=$(ls -td tests/results/e2e_suite_* | head -1)
cat "$RDIR/summary.txt"
```

### CUDA E2E troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| SFT/RM/DPO tests all exit before a `loss=` line with `ModuleNotFoundError: No module named 'mpi4py'` | The direct DeepSpeed supervised path needs the MPI Python binding. | Install and import-check `mpi4py` as in Part 2. |
| `vlm_grpo` raises `Repo id must be in the form ... /tmp/hf_vlm_clean/Qwen2-VL-2B-Instruct` | The configured local VLM directory does not exist. | Run the `snapshot_download` command above, then rerun `bash tests/test_e2e_suite_multigpu.sh vlm_grpo`. |
| `DeepSpeed Op Builder: Installed CUDA version ... does not match` | CUDA toolkit and PyTorch build versions differ. | `export DS_SKIP_CUDA_CHECK=1`; the suite exports it automatically on CUDA. |
| Ray reports `/tmp/ray/... is over 95% full` | Ray temporary data has consumed the container filesystem. | Stop Ray, remove stale `/tmp/ray` and `/tmp/e2e_*` directories, then confirm free disk space before retrying. |

The optional Triton-kernel import warning from vLLM is not by itself a test failure if the test
continues to produce `Global step` lines and the suite records `PASS`.

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
- [ ] (Part 7) For multi-GPU E2E: expose two free GPUs, install `mpi4py`, verify two CUDA
  devices, and pre-download `Qwen/Qwen2-VL-2B-Instruct`
- [ ] (Part 7) Confirm the smoke banner says `Device: cuda | Sync backend: nccl` before the
  full E2E suite
