# Starting / Resuming the NVIDIA OpenRLHF Docker Container

Quick-reference for day-to-day use, once the environment has been set up the first time via
`SETUP_new_machine_nvidia_generic.md`. Use **Case A** every time after the first — it resumes
the exact same container with everything already installed, no reinstalling needed.

---

## Case A — Resuming a previously-created named container (normal case)

This only works if the container was created with `--name` (not `--rm`) the first time. Check
whether it still exists:
```bash
docker ps -a --filter "name=openrlhf-nvidia"
```
If it's listed (even as `Exited`), resume it:
```bash
docker start -ai openrlhf-nvidia
```
This drops you back into the container's shell with everything you previously installed
(`openrlhf[vllm]`, `flash-attn`, `pytest`, etc.) still intact — no reinstalling anything.

You'll land back wherever the container's working directory was when it stopped. `cd` to your
OpenRLHF checkout if needed:
```bash
find / -maxdepth 5 -iname "test_ray_env_vars.py" 2>/dev/null   # if unsure where it landed
```

**To exit without destroying the container** (so Case A works again next time), just type
`exit` — as long as the container was NOT started with `--rm`, exiting only stops it, it does
not delete it.

---

## Case B — Starting fresh (first time, or the old container was removed / used `--rm`)

**Before running `docker run`, `cd` into the exact OpenRLHF checkout folder you want mounted**
— the `-v $PWD:/openrlhf` flag captures your current directory at the moment you run the
command:
```bash
cd ~/madhu/openrlhf_exp/OpenRLHF   # adjust to your actual path
```

**If this host needs a proxy for internet access**, check first:
```bash
cat /etc/environment | grep -i proxy
```
Proxy variables set there are NOT automatically inherited by Docker — they must be passed in
explicitly with `-e`. Fill in the actual proxy URL/port from that check:
```bash
docker run --runtime=nvidia -it --name openrlhf-nvidia --shm-size="10g" --cap-add=SYS_ADMIN \
  -e http_proxy="http://<proxy-host>:<port>" \
  -e https_proxy="http://<proxy-host>:<port>" \
  -e HTTP_PROXY="http://<proxy-host>:<port>" \
  -e HTTPS_PROXY="http://<proxy-host>:<port>" \
  -e no_proxy="<your no_proxy value>" \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:26.03-py3 bash
```
If no proxy is needed on this host, drop the 5 `-e ..._proxy` lines.

**`--name openrlhf-nvidia` is deliberate, not `--rm`** — this is what makes Case A possible
next time. If you use `--rm` here, the container is destroyed on exit and you must redo every
install below, every session.

Once inside (as `root`), verify network access before installing anything:
```bash
python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"
```
Should print `200`. If it raises `OSError: [Errno 101] Network is unreachable`, the proxy
wasn't passed in — exit (`exit`), remove the failed container (`docker rm openrlhf-nvidia`),
and retry `docker run` with the correct proxy values.

Then run the full install sequence:
```bash
pip uninstall xgboost transformer_engine flash_attn pynvml -y
pip install flash-attn==2.8.3 --no-build-isolation
pip install openrlhf[vllm]
pip install pytest
```

From here on, use **Case A** to come back to this exact container.

---

## Verifying everything still works (either case)

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
"
nvidia-smi
```

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B', dtype='bfloat16', gpu_memory_utilization=0.5, enforce_eager=True)
outputs = llm.generate(['The capital of France is'], SamplingParams(max_tokens=10, temperature=0))
print(outputs[0].outputs[0].text)
"
```

```bash
python -c "import ray; ray.init(); print(ray.cluster_resources())"
```
Should show a `GPU` key automatically (no manual `--num-gpus` needed on NVIDIA).

```bash
cd <your OpenRLHF checkout path inside the container>
python -m pytest tests/ -v
```

---

## Cleaning up (only if you actually want to delete everything)

```bash
docker stop openrlhf-nvidia
docker rm openrlhf-nvidia
```
This permanently deletes the container and everything installed inside it. Your host files
under the mounted directory (e.g. source code, test files) are untouched — only the container's
own installed packages are lost. After this, you must use Case B (fresh start) next time.
