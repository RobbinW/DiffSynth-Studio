"""Bounded native H3 LoRA training; validation generates complete clips."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor, distribute_tensor

from .config import Config
from .distributed import average_gradients, clip_gradients, primary, setup
from .encode import Decoder
from .media import read_index, read_video, resample, write_video
from .model import adapter_parameters, load_model, without_adapter
from .runtime import NativeRuntime, move, training_inputs
from .sampling import Samples, validation_rows
from .temporal import valid_latents
from .trajectories import audio_trajectory


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
    values = {name: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu().contiguous()
              for name, value in adapter_parameters(model).items()}
    # PyTorch initializes empty optimizer state by taking a zero-LR step.
    # Keep step0 truly untouched; collect state only after a real update.
    optimizer_state = get_optimizer_state_dict(model, optimizer, options=StateDictOptions(
        full_state_dict=True, cpu_offload=True)) if update else None
    if primary():
        folder = out / f"step_{update:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        save_file(values,
                  str(folder / "adapter.safetensors"), metadata={"rank": str(config.rank), "alpha": str(config.rank)})
        torch.save({"update": update, "optimizer": optimizer_state, "rng": states}, folder / "training.pt")
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
    device, group, shard_group, dp_rank, dp_size = setup(config.sequence_parallel)
    torch.manual_seed(config.seed)
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized(), shard_group=shard_group)
    parameters = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_index(cache)
    samples = Samples([row for row in rows if row["split"] == "train"])
    validation = validation_rows(rows)
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    leader = not dist.is_initialized() or dist.get_rank(group) == 0
    decoder = Decoder(args.model, device) if leader else None
    start = 0
    if args.resume:
        folder = Path(args.resume)
        values = load_file(str(folder / "adapter.safetensors"))
        for name, parameter in adapter_parameters(model).items():
            value = values[name].to(device)
            if isinstance(parameter, DTensor):
                value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
            parameter.data.copy_(value)
        state = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
        if state["optimizer"] is not None:
            set_optimizer_state_dict(model, optimizer, state["optimizer"], options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=False))
        start = state["update"]
        rng = state["rng"][dist.get_rank() if dist.is_initialized() else 0]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
    if primary():
        config.save(out / "config.json")
        print(json.dumps({"phase": "baseline_validation", "world": dp_size * config.sequence_parallel,
                          "train_windows": sum(row["split"] == "train" for row in rows),
                          "validation_windows": len(validation), "steps": config.steps}), flush=True)
    validate(model, runtime, decoder, validation, cache, out, start, dp_rank, dp_size, config)
    if start == 0:
        save_checkpoint(model, optimizer, 0, out, config)
    for update in range(start + 1, config.steps + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config.accumulation):
            rng = random.Random(config.seed + (update * config.accumulation + micro) * dp_size + dp_rank)
            row = samples.choose(rng)
            sample = torch.load(cache / row["file"], map_location="cpu", weights_only=False)
            seed = rng.choice(config.trajectory_seeds)
            trajectory = audio_trajectory(model, runtime, sample, cache / row["file"], seed, group)
            prepared = move(sample["prepared"][seed], device)
            index = rng.randrange(config.inference_steps)
            audio = trajectory[index].to(device)
            video, audio, target = training_inputs(prepared, sample["target"].to(device), audio, index)
            teacher = None
            if config.preserve_weight:
                with torch.no_grad(), without_adapter(model):
                    teacher = runtime.predict(model, prepared, video, audio, index)[0][0]
            torch.cuda.synchronize()
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                prediction = runtime.predict(model, prepared, video, audio, index)[0][0]
            torch.cuda.synchronize()
            nv = prepared["num_condition_video_rows"]
            loss = video_loss(prediction[nv:], target, sample["metadata"])
            if teacher is not None:
                loss = loss + config.preserve_weight * video_loss(prediction[nv:], teacher[nv:], sample["metadata"])
            (loss / config.accumulation).backward()
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            del prediction, loss, prepared, video, audio, target, trajectory, sample, teacher
        average_gradients(parameters)
        torch.cuda.synchronize()
        norm = clip_gradients(parameters, config.gradient_clip, shard_group)
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.lr(update)
        optimizer.step()
        torch.cuda.synchronize()
        mean_loss = torch.tensor(sum(losses) / len(losses), device=device)
        if dist.is_initialized():
            dist.all_reduce(mean_loss)
            mean_loss.div_(dist.get_world_size())
        if primary():
            record = {"update": update, "loss": float(mean_loss), "grad_norm": float(norm),
                      "lr": config.lr(update), "seconds": time.monotonic() - started}
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
