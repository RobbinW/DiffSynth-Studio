"""One contiguous sequence/weight-shard group per node."""

import os

import torch
import torch.distributed as dist


def setup(size):
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if "RANK" not in os.environ:
        return device, None, 0, 1
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % size:
        raise ValueError("World size must be divisible by sequence_parallel")
    selected = None
    for start in range(0, world, size):
        group = dist.new_group(list(range(start, start + size)))
        if start <= rank < start + size:
            selected = group
    return device, selected, rank // size, world // size


def average_gradients(parameters):
    if not dist.is_initialized():
        return
    for parameter in parameters:
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(dist.get_world_size())


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0
