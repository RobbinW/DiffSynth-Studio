"""torchrun --standalone --nproc_per_node2 -m ...tests.check_parallel"""

import copy
import os

import torch
import torch.distributed as dist
from diffusers import MiniMaxH3Transformer3DModel

from ..attention import SequenceParallel
from ..model import adapter_parameters, add_adapter


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.new_group(list(range(dist.get_world_size())))
    torch.manual_seed(42)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=4, attention_head_dim=16,
                                      hidden_size=64, num_layers=2, num_refiner_layers=1,
                                      ffn_dim=128, text_dim=32, freq_dim=16,
                                      time_embed_hidden_dim=64, time_embed_dim=32, rope_freq_dim=2)
    add_adapter(model, 2)
    with torch.no_grad():
        for name, parameter in adapter_parameters(model).items():
            if "lora_B" in name:
                parameter.normal_(std=0.01)
    reference = copy.deepcopy(model).cuda()
    parallel = model.cuda()
    SequenceParallel(reference)
    SequenceParallel(parallel, group)
    inputs = dict(hidden_states=torch.randn(1, 11, 96, device="cuda"),
                  audio_hidden_states=torch.randn(1, 7, 32, device="cuda"),
                  encoder_hidden_states=torch.randn(1, 5, 32, device="cuda"),
                  position_ids=torch.randn(23, 3, dtype=torch.float64, device="cuda"),
                  token_tags=torch.tensor([1] * 5 + [0] * 11 + [2] * 7, device="cuda"),
                  text_indices=torch.arange(5, device="cuda"), video_indices=torch.arange(5, 16, device="cuda"),
                  audio_indices=torch.arange(16, 23, device="cuda"), timestep=torch.tensor([0.3, 0.9], device="cuda"),
                  timestep_indices=torch.arange(23, device="cuda") % 2, return_dict=False)
    expected = reference(**inputs)
    sum(value.square().mean() for value in expected).backward()
    for checkpoint in (False, True):
        parallel.zero_grad(set_to_none=True)
        if checkpoint:
            parallel.enable_gradient_checkpointing()
        actual = parallel(**inputs)
        sum(value.square().mean() for value in actual).backward()
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
        for name, parameter in adapter_parameters(parallel).items():
            dist.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(dist.get_world_size(group))
            torch.testing.assert_close(parameter.grad, adapter_parameters(reference)[name].grad,
                                       atol=2e-5, rtol=2e-3)
    if dist.get_rank() == 0:
        print("Uneven SP forward/backward and activation recomputation match the unsharded reference.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
