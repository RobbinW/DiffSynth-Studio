"""Original mixed-precision H3 plus attention-only FP32 task adapters."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers import transformer_minimax_h3 as h3
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from .attention import SequenceParallel


def parameter_dtype(module):
    # FSDP recomputation exposes Linear.weight as a Tensor view, not a registered
    # Parameter. Diffusers' Python3.10 fallback also shadows the builtin tuple.
    return module.weight.dtype if hasattr(module, "weight") else next(module.parameters()).dtype


h3.get_parameter_dtype = parameter_dtype


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.enabled = True

    def forward(self, hidden):
        output = self.base(hidden)
        if self.enabled:
            output = output + self.lora_B(self.lora_A(hidden.to(self.lora_A.weight.dtype))).to(output.dtype)
        return output


def adapter_parameters(model):
    return {name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): value
            for name, value in model.named_parameters() if "lora_" in name}


@contextmanager
def without_adapter(model):
    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    for module in adapters:
        module.enabled = False
    try:
        yield
    finally:
        for module in adapters:
            module.enabled = True


def add_adapter(model, rank):
    model.requires_grad_(False)
    for block in model.transformer_blocks:
        for name in ("to_q", "to_k", "to_v"):
            setattr(block.attn, name, LoRALinear(getattr(block.attn, name), rank))
        block.attn.to_out[0] = LoRALinear(block.attn.to_out[0], rank)
    return model


def load_model(path, device, rank=32, group=None, shard=False, shard_group=None):
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        path, subfolder="transformer_ref", torch_dtype=torch.bfloat16, local_files_only=True
    )
    for name, parameter in model.named_parameters():
        expected = torch.float32 if any(owner in name for owner in model._keep_in_fp32_modules) else torch.bfloat16
        if parameter.dtype != expected:
            raise ValueError(f"Unexpected checkpoint dtype: {name} {parameter.dtype}, expected {expected}")
    add_adapter(model, rank)
    SequenceParallel(model, group)
    if not shard:
        return model.to(device)
    mesh = DeviceMesh.from_group(shard_group, "cuda")
    # FSDP2 owns the adapters and supports the checkpoint's mixed dtypes directly.
    # Move/shard one block at a time; the full33B model never resides on one card.
    for blocks in (model.transformer_blocks, model.token_refiner.refiner_blocks):
        for block in blocks:
            block.to(device)
            fully_shard(block, mesh=mesh)
    model.to(device)
    fully_shard(model, mesh=mesh)
    return model
