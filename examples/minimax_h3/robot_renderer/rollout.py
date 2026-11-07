"""Generate158 frames per call and reuse the generated boundary as next reference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor, distribute_tensor

from .config import Config
from .distributed import setup
from .encode import Decoder, Encoder
from .media import read_manifest, read_video, resample, write_video
from .model import adapter_parameters, load_model
from .runtime import NativeRuntime, move
from .temporal import pad_tail, rollout_windows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, shard_group, dp_rank, dp_size = setup(config.sequence_parallel)
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized(), shard_group=shard_group).eval()
    if args.adapter:
        values = load_file(args.adapter)
        for name, parameter in adapter_parameters(model).items():
            parameter.data.copy_(distribute_tensor(values[name].to(device), parameter.device_mesh, parameter.placements)
                                 if isinstance(parameter, DTensor) else values[name].to(device))
    leader = not dist.is_initialized() or dist.get_rank(group) == 0
    encoder = Encoder(args.model, str(device)) if leader else None
    decoder = Decoder(args.model, device) if leader else None
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for row in read_manifest(args.manifest)[dp_rank::dp_size]:
        if leader:
            source, pts = read_video(row["proxy"], row["width"], row["height"])
            times = np.arange(int(np.floor(pts[-1] * config.fps + 1e-6)) + 1) / config.fps
            proxy = resample(source, pts, times)
            reference = Image.open(row["reference"]).convert("RGB")
            output, segments = [], []
        total = [len(proxy) if leader else None]
        source_rank = dp_rank * config.sequence_parallel
        if dist.is_initialized():
            dist.broadcast_object_list(total, src=source_rank, group=group)
        for segment, window in enumerate(rollout_windows(total[0], config.frames)):
            metadata = dict(row, frames=config.frames)
            prepared = [None]
            if leader:
                clip = pad_tail(proxy[window.start:window.start + window.valid], config.frames)
                raw = encoder.conditions(reference, clip, metadata, row["prompt"])
                prepared[0] = move(runtime.prepare(raw, metadata, config.seed + segment), "cpu")
            if dist.is_initialized():
                dist.broadcast_object_list(prepared, src=source_rank, group=group)
            latents, _ = runtime.sample(model, prepared[0], metadata)
            if leader:
                frames = decoder(latents)
                reference = frames[window.valid - 1]
                output.extend(frames[int(segment > 0):window.valid])
                segments.append({"start": window.start, "valid": window.valid, "generated": config.frames})
        if leader:
            write_video(output, out / f"{row['id']}.mp4", config.fps)
            (out / f"{row['id']}.json").write_text(json.dumps({"frames": len(output), "segments": segments}, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
