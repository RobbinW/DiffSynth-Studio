"""Original mixed-precision H3 plus attention-only FP32 task adapters."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from diffusers import MiniMaxH3Transformer3DModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from .attention import SequenceParallel


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
            output = output + self.lora_B(self.lora_A(hidden.float())).to(output.dtype)
        return output


def adapter_parameters(model):
    return {name.replace("_fsdp_wrapped_module.", ""): value
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


def load_model(path, device, rank=32, group=None, shard=False):
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        path, subfolder="transformer_ref", torch_dtype=torch.bfloat16, local_files_only=True
    )
    for name, parameter in model.named_parameters():
        expected = torch.float32 if any(owner in name for owner in model._keep_in_fp32_modules) else torch.bfloat16
        if parameter.dtype != expected:
            raise ValueError(f"Unexpected checkpoint dtype: {name} {parameter.dtype}, expected {expected}")
    add_adapter(model, rank)
    SequenceParallel(model, group)
    model.enable_gradient_checkpointing()
    if not shard:
        return model.to(device)
    ignored = [parameter for parameter in model.parameters() if parameter.dtype == torch.float32]
    for parameter in ignored:
        parameter.data = parameter.data.to(device)

    def wrap(module, recurse, nonwrapped_numel):
        return recurse or type(module).__name__ in {"MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"}

    return FSDP(model, process_group=group, auto_wrap_policy=wrap, device_id=device,
                ignored_states=ignored, use_orig_params=True,
                sharding_strategy=ShardingStrategy.FULL_SHARD, limit_all_gathers=True)
