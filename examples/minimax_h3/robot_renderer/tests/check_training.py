"""Distributed optimizer and checkpoint parity against an unsharded reference."""

import argparse
import copy
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

from ..config import Config
from ..distributed import setup, average_gradients, clip_gradients, primary
from ..model import LoRALinear, adapter_parameters
from ..train import save_checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Linear(16, 16, dtype=torch.bfloat16).requires_grad_(False)
        self.projection = LoRALinear(base, 32)

    def forward(self, hidden):
        return self.projection(hidden).float()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sequence-parallel', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    device, group, shard_group, dp_rank, dp_size = setup(args.sequence_parallel)
    torch.manual_seed(42)
    model = Tiny().to(device)
    reference = copy.deepcopy(model)
    fully_shard(model, mesh=DeviceMesh.from_group(shard_group, 'cuda'))
    parameters = list(adapter_parameters(model).values())
    reference_parameters = list(adapter_parameters(reference).values())
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, betas=(0.9, 0.95))
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=1e-3, betas=(0.9, 0.95))
    out = Path(args.out)
    save_checkpoint(model, optimizer, 0, out, Config())

    def batch(update, micro, rank):
        generator = torch.Generator(device=device).manual_seed(update * 100 + micro * 10 + rank)
        return (torch.randn(1, 8, 16, device=device, dtype=torch.bfloat16, generator=generator),
                torch.randn(1, 8, 16, device=device, generator=generator))

    for update in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            hidden, target = batch(update, micro, dp_rank)
            ((model(hidden) - target).square().mean() / 2).backward()
            for rank in range(dp_size):
                hidden, target = batch(update, micro, rank)
                ((reference(hidden) - target).square().mean() / (2 * dp_size)).backward()
        average_gradients(parameters)
        for actual, expected in zip(parameters, reference_parameters):
            torch.testing.assert_close(actual.grad.full_tensor(), expected.grad, atol=2e-7, rtol=2e-5)
        norm = clip_gradients(parameters, 1, shard_group)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference_parameters, 1)
        torch.testing.assert_close(norm, expected_norm, atol=2e-7, rtol=2e-5)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(parameters, reference_parameters):
            torch.testing.assert_close(actual.full_tensor(), expected, atol=2e-7, rtol=2e-5)
        if update == 2:
            save_checkpoint(model, optimizer, update, out, Config())
            payload = [None]
            if primary():
                folder = out / 'step_000002'
                payload[0] = (load_file(str(folder / 'adapter.safetensors')),
                              torch.load(folder / 'training.pt', weights_only=False))
            dist.broadcast_object_list(payload, src=0)
            values, state = payload[0]
            with torch.no_grad():
                for name, parameter in adapter_parameters(model).items():
                    parameter.copy_(distribute_tensor(values[name].to(device), parameter.device_mesh,
                                                      parameter.placements))
            optimizer = torch.optim.AdamW(parameters, lr=1e-3, betas=(0.9, 0.95))
            optimizer.zero_grad(set_to_none=True)
            set_optimizer_state_dict(model, optimizer, state['optimizer'], options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=False))
        if primary():
            print(f'PASS update={update} world={dist.get_world_size()} dp={dp_size}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
