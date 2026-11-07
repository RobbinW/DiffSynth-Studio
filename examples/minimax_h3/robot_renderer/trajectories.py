"""Cache frozen native audio states; never substitute noised silence in training."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from .config import Config
from .distributed import setup
from .model import load_model, without_adapter
from .runtime import NativeRuntime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, dp_rank, dp_size = setup(config.sequence_parallel)
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized()).eval()
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    paths = sorted(Path(args.cache).glob("*.pt"))
    with without_adapter(model):
        for path in paths[dp_rank::dp_size]:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            trajectories = {}
            for seed in config.trajectory_seeds:
                baseline, audio = runtime.sample(model, sample["prepared"][seed], sample["metadata"], record_audio=True)
                trajectories[seed] = {"audio": torch.stack(audio), "baseline": baseline.cpu()}
            if not dist.is_initialized() or dist.get_rank(group) == 0:
                torch.save(trajectories, path.with_suffix(".audio"))
                print(path, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
