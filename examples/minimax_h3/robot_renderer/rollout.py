"""Full autoregressive evaluation with lossless boundary references and resumable clips."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

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
from .metrics import PROTOCOL, atomic_json, evaluate_file
from .model import adapter_parameters, load_model
from .runtime import NativeRuntime, move
from .temporal import pad_tail, rollout_windows


def image_hash(image):
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def stitch(folder, fps):
    """Use the same encoded segment files on fresh and resumed runs."""
    output = []
    for index, path in enumerate(sorted((folder / "segments").glob("*/segment.json"))):
        item = json.loads(path.read_text())
        frames, _ = read_video(path.with_name("generated.mp4"), item["width"], item["height"])
        output.extend(frames[int(index > 0):item["valid"]])
    write_video(output, folder / "generated.mp4", fps)
    return len(output)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--variants", help="JSON object mapping names to adapter paths; null means original base")
    parser.add_argument("--config")
    parser.add_argument("--encoder-devices", nargs="+", help="Dedicated visible CUDA devices, e.g. cuda:2 cuda:3")
    parser.add_argument("--memory-limit-gib", type=float, default=60)
    parser.add_argument("--memory-budgets", help="JSON rank -> visible device -> GiB, for shared conditioner cards")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, shard_group, dp_rank, dp_size = setup(config.sequence_parallel)
    torch.set_num_threads(4)
    leader = not dist.is_initialized() or dist.get_rank(group) == 0
    encoder_devices = args.encoder_devices or [str(device)]
    owned_devices = {device.index} | ({torch.device(x).index for x in encoder_devices} if leader else set())
    rank = dist.get_rank() if dist.is_initialized() else 0
    budgets = (json.loads(Path(args.memory_budgets).read_text())[str(rank)] if args.memory_budgets
               else {str(index): args.memory_limit_gib for index in owned_devices})
    for index in owned_devices:
        torch.cuda.set_per_process_memory_fraction(
            min(1., budgets[str(index)] * 1024**3 / torch.cuda.get_device_properties(index).total_memory), index)
    source_rank = dp_rank * config.sequence_parallel
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        if leader:
            state = dict(phase=phase, time=time.time(), **extra)
            atomic_json(out / f"status_dp{dp_rank}.json", state)
            print(json.dumps(state), flush=True)
    status("loading_transformer")
    model = load_model(args.model, device, config.rank, group, shard=dist.is_initialized(),
                       shard_group=shard_group, attention_backend=config.attention_backend).eval()
    model.requires_grad_(False)
    parameters = adapter_parameters(model)
    status("loading_encoder")
    encoder = Encoder(args.model, encoder_devices[0], args.encoder_devices) if leader else None
    decoder = Decoder(args.model, device) if leader else None
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    variants = json.loads(Path(args.variants).read_text()) if args.variants else {"adapter" if args.adapter else "base": args.adapter}
    # Spawn avoids forking an initialized CUDA runtime. Metrics use CPU video decoding only.
    evaluator = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) if leader else None
    metric_tasks = []
    def queue_metrics(path):
        if json.loads(path.read_text()).get("metric_protocol") != PROTOCOL:
            metric_tasks.append(evaluator.submit(evaluate_file, path, config.fps))
    for row in read_manifest(args.manifest)[dp_rank::dp_size]:
        if leader:
            source, pts = read_video(row["proxy"], row["width"], row["height"])
            times = np.arange(int(np.floor(pts[-1] * config.fps + 1e-6)) + 1) / config.fps
            proxy = resample(source, pts, times)
        total = [len(proxy) if leader else None]
        if dist.is_initialized():
            dist.broadcast_object_list(total, src=source_rank, group=group)
        for variant, adapter in variants.items():
            folder = out / row["id"] / variant
            complete = [bool((folder / "result.json").exists()) if leader else None]
            if dist.is_initialized():
                dist.broadcast_object_list(complete, src=source_rank, group=group)
            if complete[0]:
                if leader and "real" in row:
                    for path in [folder / "metrics.json", *sorted((folder / "segments").glob("*/metrics.json"))]:
                        queue_metrics(path)
                continue
            state = load_file(adapter) if adapter else {}
            if adapter and set(state) != set(parameters):
                raise ValueError("Checkpoint parameter names do not match the current task adapter")
            for name, parameter in parameters.items():
                if adapter:
                    value = state[name].to(device)
                    parameter.copy_(distribute_tensor(value, parameter.device_mesh, parameter.placements)
                                    if isinstance(parameter, DTensor) else value)
                else:
                    parameter.zero_()
            del state
            if leader:
                reference = Image.open(row["reference"]).convert("RGB")
            started = time.time()
            windows = rollout_windows(total[0], config.frames)
            for segment, window in enumerate(windows):
                child = folder / "segments" / f"{segment:06d}"
                complete = [bool((child / "segment.json").exists()) if leader else None]
                if dist.is_initialized():
                    dist.broadcast_object_list(complete, src=source_rank, group=group)
                if complete[0]:
                    if leader:
                        previous = json.loads((child / "segment.json").read_text())
                        if previous["reference_sha256"] != image_hash(reference):
                            raise ValueError("Saved segment does not continue the generated reference chain")
                        reference = Image.open(child / "last_frame.png").convert("RGB")
                        if previous["last_frame_sha256"] != image_hash(reference):
                            raise ValueError("Saved boundary image hash mismatch")
                        if "real" in row:
                            queue_metrics(child / "metrics.json")
                    continue
                metadata = dict(row, frames=config.frames, start=window.start, valid=window.valid)
                prepared = [None]
                status("encoding", id=row["id"], variant=variant, segment=segment, total_segments=len(windows))
                segment_started = time.time()
                if leader:
                    child.mkdir(parents=True, exist_ok=True)
                    reference.save(child / "reference.png")
                    reference_sha = image_hash(reference)
                    clip = pad_tail(proxy[window.start:window.start + window.valid], config.frames)
                    raw = encoder.conditions(reference, clip, metadata, row["prompt"])
                    prepared[0] = move(runtime.prepare(raw, metadata, config.seed + segment), "cpu")
                    del raw
                if dist.is_initialized():
                    dist.broadcast_object_list(prepared, src=source_rank, group=group)
                status("sampling", id=row["id"], variant=variant, segment=segment, total_segments=len(windows))
                latent, _ = runtime.sample(model, prepared[0], metadata)
                if leader:
                    frames = decoder(latent)[:window.valid]
                    reference = frames[-1]
                    reference.save(child / "last_frame.png")
                    write_video(frames, child / "generated.mp4", config.fps)
                    atomic_json(child / "metrics.json", dict(seed=config.seed + segment, metadata=metadata))
                    atomic_json(child / "segment.json", dict(start=window.start, valid=window.valid,
                        generated=config.frames, width=row["width"], height=row["height"],
                        reference_source="real_first_frame" if segment == 0 else "previous_generated_last_frame",
                        reference_sha256=reference_sha, last_frame_sha256=image_hash(reference),
                        seconds=time.time()-segment_started, seed=config.seed + segment))
                    if "real" in row:
                        queue_metrics(child / "metrics.json")
                    status("segment_complete", id=row["id"], variant=variant, segment=segment,
                           total_segments=len(windows), seconds=time.time()-segment_started)
                    del frames
                del latent, prepared
                peaks = {str(i): dict(allocated_gib=torch.cuda.max_memory_allocated(i) / 1024**3,
                                      reserved_gib=torch.cuda.max_memory_reserved(i) / 1024**3)
                         for i in owned_devices}
                all_peaks = [peaks]
                if dist.is_initialized():
                    all_peaks = [None] * dist.get_world_size(group)
                    dist.all_gather_object(all_peaks, peaks, group=group)
                if leader:
                    atomic_json(folder / "memory_peaks.json", all_peaks)
            if leader:
                count = stitch(folder, config.fps)
                assert count == total[0]
                atomic_json(folder / "metrics.json", dict(seed=config.seed,
                    metadata=dict(row, start=0, valid=count, frames=count)))
                if "real" in row:
                    queue_metrics(folder / "metrics.json")
                atomic_json(folder / "result.json", dict(id=row["id"], variant=variant, adapter=adapter,
                    frames=count, fps=config.fps, segments=len(windows), clip_frames=config.frames,
                    reference_policy="generated_last_frame", overlap_frames=1, remove_overlap_on_stitch=True,
                    inference_steps=config.inference_steps, decfg_inference=False, seconds=time.time()-started))
                status("rollout_complete", id=row["id"], variant=variant, frames=count)
    if leader:
        for future in metric_tasks:
            future.result()
        evaluator.shutdown()
    if dist.is_initialized():
        dist.barrier()
    status("complete")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
