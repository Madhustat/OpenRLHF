# OpenRLHF XCCL validation image
# Base: vllm/vllm-openai-xpu:v0.27.1
#   torch 2.13.0+xpu | vLLM 0.27.1 | Intel runtime 2026.0.0 | oneCCL 2022.0.0
#
# Layers added:
#   1. OpenRLHF training deps (deepspeed, peft, ...)
#   2. vLLM PR #52389 — world_size>1 guard on XPU warm-up all_reduce
#   3. DeepSpeed AdamW fallback — icpx not present in this serving image
FROM vllm/vllm-openai-xpu:v0.27.1

# Training libraries.
# --no-deps protects the synced torch 2.13 / vLLM 0.27 / Intel-runtime stack.
RUN pip install --no-cache-dir --no-deps deepspeed==0.19.5 peft==0.20.0 loralib && \
    pip install --no-cache-dir \
        hjson ninja py-cpuinfo msgpack nvidia-ml-py \
        jsonlines pylatexenc transformers_stream_generator optree \
        accelerate torchdata torchmetrics

# ninja on system PATH so Ray worker subprocesses (invoked via /bin/sh) can find it.
RUN ln -sf /opt/venv/bin/ninja /usr/local/bin/ninja

# Patch 1: vLLM PR #52389 — skip world_size=1 oneCCL warm-up all_reduce.
COPY scripts/patch_vllm_xpu_worker.py /tmp/patch_vllm_xpu_worker.py
RUN python /tmp/patch_vllm_xpu_worker.py

# Patch 2: DeepSpeed AdamW fallback when icpx JIT compile is unavailable.
COPY scripts/patch_ds_adam.py /tmp/patch_ds_adam.py
RUN python /tmp/patch_ds_adam.py
