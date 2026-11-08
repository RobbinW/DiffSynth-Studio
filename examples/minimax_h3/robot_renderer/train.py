"""Native H3 LoRA training; validation generates complete clips."""

import argparse
import json
from pathlib import Path
import random
import signal
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
from .decfg import decfg_mode
from .distributed import average_gradients, clip_gradients, primary, setup
from .encode import Decoder
from .media import read_index, read_video, resample, write_video
from .model import adapter_parameters, configure_training_memory, load_model
from .runtime import NativeRuntime, move, training_inputs
from .sampling import Samples, validation_rows
from .temporal import valid_latents


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
        (folder / "adapter.safetensors").chmod(0o644)
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
    parser.add_argument("--skip-initial-validation", action="store_true",
                        help="Resume a saved update without repeating validation; scheduled checks still run")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, shard_group, dp_rank, dp_size = setup(config.sequence_parallel)
    torch.manual_seed(config.seed)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def stopping():
        flag = torch.tensor(int(stop_requested), device=device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized(),
                       shard_group=shard_group, decfg_path=config.decfg_path,
                       attention_backend=config.attention_backend)
    configure_training_memory(model, config.checkpoint_stride)
    parameters = list(adapter_parameters(model).values())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 betas=(0.9, 0.95), eps=1e-8, weight_decay=config.weight_decay)
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_index(cache)
    samples = Samples([row for row in rows if row["split"] == "train"])
    validation = validation_rows(rows)
    silence = torch.load(cache / "silence.pt", map_location=device, weights_only=True)
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
        optimizer_steps = [float(s["step"]) for s in optimizer.state.values()]
        assert optimizer_steps and min(optimizer_steps) == max(optimizer_steps) == start
        if primary():
            print(json.dumps({"phase": "resumed", "update": start, "optimizer_step": min(optimizer_steps),
                              "rng_restored": True, "audio_conditioning": "noised_silence"}), flush=True)
    if primary():
        config.save(out / "config.json")
        print(json.dumps({"phase": "ready", "world": dp_size * config.sequence_parallel,
                          "train_windows": sum(row["split"] == "train" for row in rows),
                          "validation_windows": len(validation), "steps": config.steps}), flush=True)
    if not (args.resume and args.skip_initial_validation):
        validate(model, runtime, decoder, validation, cache, out, start, dp_rank, dp_size, config)
    if start == 0:
        save_checkpoint(model, optimizer, 0, out, config)
    update = start
    model.train()
    while (config.steps is None or update < config.steps) and not stopping():
        update += 1
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        phases = dict(input=0.0, forward=0.0, backward=0.0, gradient_sync=0.0, optimizer=0.0)
        used = []
        for micro in range(config.accumulation):
            begin = time.monotonic()
            rng = random.Random(config.seed + (update * config.accumulation + micro) * dp_size + dp_rank)
            row = samples.choose(rng)
            sample = torch.load(cache / row["file"], map_location="cpu", weights_only=False)
            seed = rng.choice(config.training_seeds)
            prepared = move(sample["prepared"][seed], device)
            index = rng.randrange(config.inference_steps)
            video, audio, target = training_inputs(prepared, sample["target"].to(device), silence[row["frames"]], index)
            torch.cuda.synchronize()
            phases["input"] += time.monotonic() - begin
            used.append(dict(id=row["id"], frames=row["frames"], seed=seed, timestep_index=index,
                             tokens=prepared["token_tags"].numel()))
            begin = time.monotonic()
            with decfg_mode(model, bool(config.decfg_path)):
                prediction = runtime.predict(model, prepared, video, audio, index)[0][0]
                torch.cuda.synchronize()
                nv = prepared["num_condition_video_rows"]
                loss = video_loss(prediction[nv:], target, sample["metadata"])
                phases["forward"] += time.monotonic() - begin
                begin = time.monotonic()
                (loss / config.accumulation).backward()
            torch.cuda.synchronize()
            phases["backward"] += time.monotonic() - begin
            losses.append(float(loss.detach()))
            del prediction, loss, prepared, video, audio, target, sample
        begin = time.monotonic()
        average_gradients(parameters)
        torch.cuda.synchronize()
        norm = clip_gradients(parameters, config.gradient_clip, shard_group)
        phases["gradient_sync"] = time.monotonic() - begin
        begin = time.monotonic()
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.lr(update)
        optimizer.step()
        torch.cuda.synchronize()
        phases["optimizer"] = time.monotonic() - begin
        mean_loss = torch.tensor(sum(losses) / len(losses), device=device)
        if dist.is_initialized():
            dist.all_reduce(mean_loss)
            mean_loss.div_(dist.get_world_size())
        if leader:
            record = {"update": update, "loss": float(mean_loss), "grad_norm": float(norm),
                      "lr": config.lr(update), "seconds": time.monotonic() - started,
                      "decfg_training_only": bool(config.decfg_path), "audio_conditioning": "noised_silence",
                      "teacher_forwards": 0, "phase_seconds": phases, "samples": used,
                      "dp_rank": dp_rank, "time": time.time(), "attention_backend": config.attention_backend,
                      "checkpoint_stride": config.checkpoint_stride, "activation_cpu_offload": False,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            with (out / ("train.jsonl" if primary() else f"train_dp{dp_rank}.jsonl")).open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if config.validate_at(update):
            save_checkpoint(model, optimizer, update, out, config)
            validate(model, runtime, decoder, validation, cache, out, update, dp_rank, dp_size, config)
    if update and not config.validate_at(update):
        save_checkpoint(model, optimizer, update, out, config)
    if primary():
        (out / "stopped.json").write_text(json.dumps({"update": update, "time": time.time(),
            "reason": "step_budget" if config.steps is not None and update >= config.steps else "stop_signal"}, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
