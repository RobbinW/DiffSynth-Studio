"""Encode sequential158-frame windows and their minimally padded tails."""

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .encode import Encoder
from .media import read_manifest, read_pair
from .runtime import NativeRuntime, move
from .temporal import training_windows, pad_tail


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    config = Config.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = Encoder(args.model, "cuda:0")
    runtime = NativeRuntime(args.model, "cpu", config.inference_steps)
    index_rows = []
    for row in read_manifest(args.manifest)[args.shard::args.shards]:
        real, proxy = read_pair(row, config.fps)
        for window in training_windows(len(real), config.frames):
            metadata = {key: row[key] for key in ("id", "source", "group", "split", "width", "height", "real", "proxy")}
            metadata.update(id=f"{row['id']}__f{window.start}", start=window.start,
                            valid=window.valid, frames=window.frames, pair_id=row["id"])
            target_path = out / f"{metadata['id']}.pt"
            real_clip = pad_tail(real[window.start:window.start + window.valid], window.frames)
            proxy_clip = pad_tail(proxy[window.start:window.start + window.valid], window.frames)
            raw = encoder.conditions(real_clip[0], proxy_clip, metadata, row["prompt"])
            prepared = {seed: runtime.prepare(move(raw, "cpu"), metadata, seed)
                        for seed in sorted(set(config.trajectory_seeds) | set(config.validation_seeds))}
            torch.save({"metadata": metadata, "target": encoder.video(real_clip, metadata).cpu(),
                        "prepared": prepared}, target_path)
            index_rows.append(dict(metadata, file=target_path.name))
            print(target_path, flush=True)
    (out / f"index.{args.shard:03d}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows))


if __name__ == "__main__":
    main()
