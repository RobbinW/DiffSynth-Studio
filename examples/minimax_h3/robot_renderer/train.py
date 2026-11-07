"""Bounded native H3 LoRA training; validation generates complete clips."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .config import Config
from .distributed import average_gradients, primary, setup
from .encode import Decoder
from .media import read_index, read_video, resample, write_video
from .model import adapter_parameters, load_model, without_adapter
from .runtime import NativeRuntime, move, training_inputs
from .temporal import valid_latents


class Samples:
    def __init__(self, rows):
        self.sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            self.sources[row["source"]][row["group"]][row["pair_id"]].append(row)

    def choose(self, rng):
        groups = self.sources[rng.choice(sorted(self.sources))]
        views = groups[rng.choice(sorted(groups))]
        return rng.choice(views[rng.choice(sorted(views))])


def video_loss(prediction, target, metadata):
    groups = torch.tensor(valid_latents(metadata["valid"], metadata["frames"]), device=prediction.device)
    rows_per_frame = (metadata["height"] // 32) * (metadata["width"] // 32)
    keep = groups.repeat_interleave(rows_per_frame)
    return F.mse_loss(prediction[keep].float(), target[keep].float())


def save_checkpoint(model, optimizer, update, out, config):
    # Sampling is a deterministic function of update/microbatch/DP rank; save all
    # device RNGs as well so future stochastic model changes still resume exactly.
    state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    states = [state]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, state)
    if primary():
        folder = out / f"step_{update:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in adapter_parameters(model).items()},
                  str(folder / "adapter.safetensors"), metadata={"rank": str(config.rank), "alpha": str(config.rank)})
        torch.save({"update": update, "optimizer": optimizer.state_dict(), "rng": states}, folder / "training.pt")
        config.save(folder / "config.json")


@torch.no_grad()
def validate(model, runtime, decoder, rows, cache, out, update, dp_rank, dp_size, config):
    model.eval()
    leader = not dist.is_initialized() or dist.get_rank() % config.sequence_parallel == 0
    for row in rows[dp_rank::dp_size]:
        sample = torch.load(cache / row["file"], map_location="cpu", weights_only=False)
        metadata = sample["metadata"]
        for seed in config.validation_seeds:
            prepared = move(sample["prepared"][seed], runtime.device)
            latent, _ = runtime.sample(model, prepared, metadata)
            if leader:
                folder = out / "validation" / f"step_{update:06d}" / row["id"] / str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                frames = decoder(latent)[:metadata["valid"]]
                write_video(frames, folder / "generated.mp4", config.fps)
                source, pts = read_video(metadata["real"], metadata["width"], metadata["height"])
                real = resample(source, pts, (metadata["start"] + np.arange(metadata["valid"])) / config.fps)
                error = (np.asarray(frames, dtype=np.float32) - np.asarray(real, dtype=np.float32)) / 255
                psnr = float(np.mean(-10 * np.log10(np.maximum((error ** 2).mean((1, 2, 3)), 1e-12))))
                (folder / "metrics.json").write_text(json.dumps({"psnr": psnr, "seed": seed, "metadata": metadata}, indent=2))
    if dist.is_initialized():
        dist.barrier()
    model.train()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, dp_rank, dp_size = setup(config.sequence_parallel)
    torch.manual_seed(config.seed)
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized())
    parameters = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_index(cache)
    samples = Samples([row for row in rows if row["split"] == "train"])
    validation = [row for row in rows if row["split"] == "val"][:dp_size * 2]
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    leader = not dist.is_initialized() or dist.get_rank(group) == 0
    decoder = Decoder(args.model, device) if leader else None
    start = 0
    if args.resume:
        folder = Path(args.resume)
        values = load_file(str(folder / "adapter.safetensors"))
        for name, parameter in adapter_parameters(model).items():
            parameter.data.copy_(values[name].to(device))
        state = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start = state["update"]
        rng = state["rng"][dist.get_rank() if dist.is_initialized() else 0]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
    if primary():
        config.save(out / "config.json")
    validate(model, runtime, decoder, validation, cache, out, start, dp_rank, dp_size, config)
    if start == 0:
        save_checkpoint(model, optimizer, 0, out, config)
    for update in range(start + 1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config.accumulation):
            rng = random.Random(config.seed + (update * config.accumulation + micro) * dp_size + dp_rank)
            row = samples.choose(rng)
            sample = torch.load(cache / row["file"], map_location="cpu", weights_only=False)
            seed = rng.choice(config.trajectory_seeds)
            trajectory = torch.load((cache / row["file"]).with_suffix(".audio"), map_location="cpu", weights_only=True)[seed]
            prepared = move(sample["prepared"][seed], device)
            index = rng.randrange(config.inference_steps)
            audio = trajectory["audio"][index].to(device)
            video, audio, target = training_inputs(prepared, sample["target"].to(device), audio, index)
            teacher = None
            if config.preserve_weight:
                with torch.no_grad(), without_adapter(model):
                    teacher = runtime.predict(model, prepared, video, audio, index)[0][0]
            prediction = runtime.predict(model, prepared, video, audio, index)[0][0]
            nv = prepared["num_condition_video_rows"]
            loss = video_loss(prediction[nv:], target, sample["metadata"])
            if teacher is not None:
                loss = loss + config.preserve_weight * video_loss(prediction[nv:], teacher[nv:], sample["metadata"])
            (loss / config.accumulation).backward()
            losses.append(float(loss.detach()))
        average_gradients(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=True)
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.lr(update)
        optimizer.step()
        if primary():
            record = {"update": update, "loss": sum(losses) / len(losses), "grad_norm": float(norm), "lr": config.lr(update)}
            with (out / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if config.validate_at(update):
            save_checkpoint(model, optimizer, update, out, config)
            validate(model, runtime, decoder, validation, cache, out, update, dp_rank, dp_size, config)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
