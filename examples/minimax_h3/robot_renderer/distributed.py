"""Separate sequence and weight-shard communicators over each node's ranks."""

import os

import torch
import torch.distributed as dist


DATA_GROUP = None


def setup(size):
    global DATA_GROUP
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if "RANK" not in os.environ:
        return device, None, None, 0, 1
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % size:
        raise ValueError("World size must be divisible by sequence_parallel")
    selected = []
    # FSDP prefetch and SP all-to-all run on different streams. Match the
    # established launcher: identical rank sets, distinct communicators.
    for _ in range(2):
        for start in range(0, world, size):
            group = dist.new_group(list(range(start, start + size)))
            if start <= rank < start + size:
                selected.append(group)
    for offset in range(size):
        group = dist.new_group(list(range(offset, world, size)))
        if rank % size == offset:
            DATA_GROUP = group
    return device, selected[0], selected[1], rank // size, world // size


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def average_gradients(parameters):
    if not dist.is_initialized() or dist.get_world_size(DATA_GROUP) == 1:
        return
    # FSDP already averaged within the SP/weight group. Average corresponding
    # local parameter shards across the four data replicas exactly once.
    flat = torch.cat([local(parameter.grad).reshape(-1) for parameter in parameters])
    dist.all_reduce(flat, group=DATA_GROUP)
    flat.div_(dist.get_world_size(DATA_GROUP))
    offset = 0
    for parameter in parameters:
        gradient = local(parameter.grad)
        count = gradient.numel()
        gradient.copy_(flat[offset:offset + count].view_as(gradient))
        offset += count


@torch.no_grad()
def clip_gradients(parameters, maximum, shard_group=None):
    norm = torch.stack([local(parameter.grad).float().square().sum() for parameter in parameters]).sum()
    if dist.is_initialized():
        dist.all_reduce(norm, group=shard_group)
    norm = norm.sqrt()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite adapter gradient norm")
    factor = float((maximum / (norm + 1e-6)).clamp(max=1))
    for parameter in parameters:
        local(parameter.grad).mul_(factor)
    return norm


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0
