def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.accelerator.synchronize()


class _GlooBroadcastCommunicator:
    """Wraps a real torch.distributed ProcessGroup (gloo backend) to present the same
    .broadcast(tensor, src) -> tensor interface as PyNcclCommunicator/StatelessProcessGroup.

    gloo does not support XPU (or other non-CPU/CUDA) tensors directly, so this stages
    through a CPU tensor. Unlike PyNcclCommunicator, this does NOT mutate its tensor
    argument in-place - callers must capture the return value, same as the
    StatelessProcessGroup fallback this replaces.
    """

    def __init__(self, pg, device):
        self.pg = pg
        self.device = device

    def broadcast(self, tensor, src):
        cpu_tensor = tensor.cpu()
        work = self.pg.broadcast(cpu_tensor, src)
        work.wait()
        return cpu_tensor.to(self.device)


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    import torch

    if torch.version.cuda is not None or torch.version.hip is not None:
        # NVIDIA/AMD: NCCL/RCCL is available, use vLLM's real GPU-to-GPU communicator.
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        return PyNcclCommunicator(pg, device=device)

    # No NCCL/RCCL on this platform (e.g. Intel XPU). PyNcclCommunicator silently sets
    # disabled=True in this case and every broadcast() becomes a no-op with no error -
    # confirmed via live instrumentation (see CORRECTION_weight_sync_not_real.md).
    #
    # First fallback tried: StatelessProcessGroup's own broadcast() (TCP-store based).
    # Correct, but each broadcast() pickles the WHOLE tensor into an in-memory TCPStore
    # entry, and under --train.colocate_all (actor + vLLM engine sharing one GPU, our
    # only option on a single-GPU machine) this measured ~2.3s per parameter - the same
    # call measured ~3-4ms in isolation without GPU colocation. A real PPO run with this
    # fallback took 809s for one weight-sync round on a 0.5B model.
    #
    # Real gloo ProcessGroup instead: a genuine torch.distributed collective backend,
    # not Python pickling through a key-value store. Verified faster in isolation
    # (~5-6ms/call) and avoids the colocation-specific stall observed with
    # StatelessProcessGroup. gloo doesn't support XPU tensors directly, so
    # _GlooBroadcastCommunicator stages through CPU.
    from vllm.distributed.utils import stateless_init_torch_distributed_process_group

    pg = stateless_init_torch_distributed_process_group(
        master_address, master_port, rank, world_size, backend="gloo"
    )
    return _GlooBroadcastCommunicator(pg, device)
