"""Cache frozen native audio states; never substitute noised silence in training."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from .config import Config
from .distributed import setup
from .model import load_model, without_adapter
from .runtime import NativeRuntime


def audio_trajectory(model, runtime, sample, path, seed, group=None):
    """Memoize the exact frozen teacher trajectory when a clip is first sampled."""
    path = Path(path).with_suffix(f".audio.{seed}.pt")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    training = model.training
    model.eval()
    with without_adapter(model):
        _, audio = runtime.sample(model, sample["prepared"][seed], sample["metadata"], record_audio=True)
    model.train(training)
    audio = torch.stack(audio)
    if not dist.is_initialized() or dist.get_rank(group) == 0:
        torch.save(audio, path)
    if dist.is_initialized():
        dist.barrier(group=group)
    return audio


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, shard_group, dp_rank, dp_size = setup(config.sequence_parallel)
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized(), shard_group=shard_group).eval()
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    paths = sorted(Path(args.cache).glob("*.pt"))
    for path in paths[dp_rank::dp_size]:
        if ".audio." in path.name:
            continue
        sample = torch.load(path, map_location="cpu", weights_only=False)
        for seed in config.trajectory_seeds:
            audio_trajectory(model, runtime, sample, path, seed, group)
        if not dist.is_initialized() or dist.get_rank(group) == 0:
            print(path, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
