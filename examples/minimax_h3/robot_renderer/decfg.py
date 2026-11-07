"""Frozen, training-only DeCFG in the native Diffusers H3 layout."""

from contextlib import contextmanager

import torch
from torch import nn
from safetensors.torch import load_file


def convert_decfg(state, heads, head_dim):
    """Convert the historical DiffSynth adapter without merging base weights.

    QKV rows are interleaved per head; SwiGLU stores gate before value.
    Only B's output rows change. A and the scale (alpha/rank = 1) stay intact.
    """
    converted, consumed = {}, set()
    suffix = ".lora_A.default.weight"
    for key in sorted(state):
        if not key.endswith(suffix):
            continue
        source = key.removesuffix(suffix)
        b_key = source + ".lora_B.default.weight"
        a, b = state[key], state[b_key]
        consumed.update((key, b_key))
        target = source.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if target.startswith("blocks."):
            target = target.replace("blocks.", "transformer_blocks.", 1)
        if target.endswith(".attn.qkv_proj"):
            grouped = b.reshape(heads, 3, head_dim, a.shape[0])
            for index, name in enumerate("qkv"):
                path = target.removesuffix("qkv_proj") + "to_" + name
                converted[path] = (a, grouped[:, index].reshape(heads * head_dim, -1).contiguous())
        elif target.endswith(".attn.out_proj"):
            converted[target.removesuffix("out_proj") + "to_out.0"] = (a, b)
        elif target.endswith(".mlp.fc1"):
            gate, value = b.chunk(2, dim=0)
            converted[target.removesuffix("mlp.fc1") + "ff.net.0.proj"] = (a, torch.cat((value, gate)))
        elif target.endswith(".mlp.fc2"):
            converted[target.removesuffix("mlp.fc2") + "ff.net.2"] = (a, b)
        else:
            raise ValueError(f"Unsupported DeCFG projection: {source}")
    if consumed != set(state):
        raise ValueError(f"Unmapped DeCFG tensors: {sorted(set(state) - consumed)}")
    return converted


class FrozenDeCFGLinear(nn.Module):
    def __init__(self, base, a, b):
        super().__init__()
        if a.shape[1] != base.in_features or b.shape != (base.out_features, a.shape[0]):
            raise ValueError("DeCFG projection shape does not match the base")
        self.base = base
        self.decfg_A = nn.Linear(base.in_features, a.shape[0], bias=False, dtype=base.weight.dtype)
        self.decfg_B = nn.Linear(a.shape[0], base.out_features, bias=False, dtype=base.weight.dtype)
        with torch.no_grad():
            self.decfg_A.weight.copy_(a)
            self.decfg_B.weight.copy_(b)
        self.decfg_A.requires_grad_(False)
        self.decfg_B.requires_grad_(False)
        self.enabled = False

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        result = self.base(hidden)
        if self.enabled:
            residual = self.decfg_B(self.decfg_A(hidden.to(self.decfg_A.weight.dtype)))
            result = result + residual.to(result.dtype)
        return result


def attach_decfg(model, path):
    from .model import LoRALinear

    projections = convert_decfg(load_file(str(path)), model.config.num_attention_heads,
                                model.config.attention_head_dim)
    expected = set()
    for prefix, blocks in (("transformer_blocks", model.transformer_blocks),
                           ("token_refiner.refiner_blocks", model.token_refiner.refiner_blocks)):
        for index in range(len(blocks)):
            expected.update(f"{prefix}.{index}.{suffix}" for suffix in
                            ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"))
    if set(projections) != expected:
        raise ValueError("DeCFG checkpoint does not cover the complete H3 trunk and text refiner")
    for name, (a, b) in projections.items():
        module = model.get_submodule(name)
        if isinstance(module, LoRALinear):
            module.base = FrozenDeCFGLinear(module.base, a, b)
        else:
            parent, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attribute, FrozenDeCFGLinear(module, a, b))


@contextmanager
def decfg_mode(model, enabled):
    adapters = [(module, module.enabled) for module in model.modules() if isinstance(module, FrozenDeCFGLinear)]
    for module, _ in adapters:
        module.enabled = enabled
    try:
        yield
    finally:
        for module, previous in adapters:
            module.enabled = previous
