"""Apply vLLM PR #52389: skip world_size=1 oneCCL warm-up all_reduce in xpu_worker.py.

vLLM 0.27.1 performs an unconditional XCCL all_reduce warm-up at world_size=1,
which hangs on some hardware. This guard skips it when there is only one worker.
Upstream issue: https://github.com/vllm-project/vllm/issues/52386
Upstream PR:    https://github.com/vllm-project/vllm/pull/52389
"""
import pathlib

f = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/xpu_worker.py")
src = f.read_text()

old = (
    "        # global all_reduce needed for overall oneccl warm up\n"
    "        if torch.distributed.is_xccl_available():\n"
    "            torch.distributed.all_reduce(torch.zeros(1).xpu())"
)
new = (
    "        # global all_reduce needed for overall oneccl warm up (PR #52389: only\n"
    "        # meaningful for multi-device runs; world_size=1 hangs on some hardware).\n"
    "        if (\n"
    "            self.parallel_config.world_size > 1\n"
    "            and torch.distributed.is_xccl_available()\n"
    "        ):\n"
    "            torch.distributed.all_reduce(torch.zeros(1).xpu())"
)

assert old in src, "target block not found — vLLM source differs from expected version"
f.write_text(src.replace(old, new, 1))
print("patched xpu_worker.py OK: added world_size > 1 guard (vLLM PR #52389)")
