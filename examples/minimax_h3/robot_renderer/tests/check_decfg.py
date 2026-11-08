"""Qualify the actual frozen DeCFG on an eight-card full33B native clip."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import distribute_tensor

from ..config import Config
from ..decfg import decfg_mode
from ..distributed import setup, clip_gradients
from ..model import adapter_parameters, configure_training_memory, load_model
from ..runtime import NativeRuntime, move, training_inputs
from ..train import video_loss


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("model", "sample", "config", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--adapter", help="Qualify an existing task checkpoint; test updates are discarded")
    args = parser.parse_args()
    config = Config.load(args.config)
    device, group, shards, _, _ = setup(config.sequence_parallel)
    torch.manual_seed(config.seed)
    model = load_model(args.model, device, config.rank, group, shard=True, shard_group=shards,
                       decfg_path=config.decfg_path, attention_backend=config.attention_backend)
    configure_training_memory(model, config.checkpoint_stride)
    model.train()
    if args.adapter:
        values = load_file(args.adapter)
        with torch.no_grad():
            for name, parameter in adapter_parameters(model).items():
                parameter.copy_(distribute_tensor(values[name].to(device), parameter.device_mesh, parameter.placements))
    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    runtime = NativeRuntime(args.model, device, config.inference_steps)
    prepared = move(sample["prepared"][42], device)
    silence = torch.load(Path(args.sample).parent / "silence.pt", map_location=device, weights_only=True)
    video, audio, target = training_inputs(prepared, sample["target"].to(device), silence[sample["metadata"]["frames"]], 25)
    parameters = list(adapter_parameters(model).values())
    frozen = {name: p.to_local().detach().clone() for name, p in model.named_parameters() if "decfg_" in name}
    assert frozen and all(not p.requires_grad for name, p in model.named_parameters() if "decfg_" in name)
    with torch.no_grad():
        baseline = runtime.predict(model, prepared, video, audio, 25)[0][0].detach()
    losses, norms, timings, reference, relative = [], [], [], None, None
    for repeat in range(3):
        started = time.monotonic()
        model.zero_grad(set_to_none=True)
        with decfg_mode(model, True):
            prediction = runtime.predict(model, prepared, video, audio, 25)[0][0]
            if repeat == 0:
                delta = float((prediction.detach().float() - baseline.float()).norm())
                assert delta > 0
            loss = video_loss(prediction[prepared["num_condition_video_rows"]:], target, sample["metadata"])
            loss.backward()
        gradient = torch.cat([p.grad.to_local().float().flatten() for p in parameters])
        norm2 = gradient.square().sum()
        dist.all_reduce(norm2, group=shards)
        assert torch.isfinite(norm2) and norm2 > 0 and torch.isfinite(loss)
        losses.append(float(loss.detach()))
        norms.append(float(norm2.sqrt()))
        if reference is not None:
            error = (reference - gradient).square().sum()
            dist.all_reduce(error, group=shards)
            relative = float((error / norm2).sqrt())
            assert relative < (1e-5 if config.attention_backend == "stable" else .05), relative
        reference = gradient.detach().clone()
        del prediction, loss, gradient
        timings.append(time.monotonic() - started)
        if dist.get_rank() == 0:
            print(json.dumps(dict(phase="backward_repeat", repeat=repeat, loss=losses[-1],
                                  norm=norms[-1], relative_l2=relative, seconds=timings[-1])), flush=True)
    stable_relative = stable_cosine = None
    if config.attention_backend != "stable":
        from ..attention import Processor
        processors = [m.processor for m in model.modules() if isinstance(getattr(m, "processor", None), Processor)]
        for processor in processors:
            processor.backend = "stable"
        configure_training_memory(model, 0)
        model.zero_grad(set_to_none=True)
        with decfg_mode(model, True):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                prediction = runtime.predict(model, prepared, video, audio, 25)[0][0]
            loss = video_loss(prediction[prepared["num_condition_video_rows"]:], target, sample["metadata"])
            loss.backward()
        stable = torch.cat([p.grad.to_local().float().flatten() for p in parameters])
        stats = torch.stack(((stable-reference).square().sum(), stable.square().sum(),
                             reference.square().sum(), (stable*reference).sum()))
        dist.all_reduce(stats, group=shards)
        stable_relative = float((stats[0]/stats[1]).sqrt())
        stable_cosine = float(stats[3]/(stats[1]*stats[2]).sqrt())
        if dist.get_rank() == 0:
            print(json.dumps(dict(phase="stable_gradient_comparison", relative_l2=stable_relative,
                                  cosine=stable_cosine)), flush=True)
        # Full-model BF16/reference comparison, not bitwise equality. Actual
        # saved-AdamW comparisons across t=0/25/49 additionally bound update error.
        assert stable_relative < .10 and stable_cosine > .995, (stable_relative, stable_cosine)
        offset = 0
        for parameter in parameters:
            gradient = parameter.grad.to_local()
            gradient.copy_(reference[offset:offset+gradient.numel()].reshape_as(gradient))
            offset += gradient.numel()
        for processor in processors:
            processor.backend = config.attention_backend
        configure_training_memory(model, config.checkpoint_stride)
        del stable, prediction, loss
    with torch.no_grad():
        restored = runtime.predict(model, prepared, video, audio, 25)[0][0]
        torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
    before = [p.to_local().detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(0.9, 0.95))
    clip_gradients(parameters, config.gradient_clip, shards)
    optimizer.step()
    changed = torch.tensor(sum(int(not torch.equal(p.to_local(), old)) for p, old in zip(parameters, before)), device=device)
    dist.all_reduce(changed)
    assert changed > 0
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter.to_local(), frozen[name], atol=0, rtol=0)
    if dist.get_rank() == 0:
        result = dict(passed=True, frames=sample["metadata"]["frames"], world=dist.get_world_size(),
                      losses=losses, gradient_norms=norms, gradient_relative_l2=relative,
                      decfg_prediction_delta_l2=delta, eval_restore_exact=True,
                      frozen_decfg_unchanged=True, task_updated=True,
                      attention_backend=config.attention_backend, backward_seconds=timings,
                      checkpoint_stride=config.checkpoint_stride, activation_cpu_offload=False,
                      acceptance=dict(repeat_relative_l2=.05, reference_relative_l2=.10, reference_cosine=.995),
                      stable_gradient_relative_l2=stable_relative, stable_gradient_cosine=stable_cosine,
                      adapter_sha256=hashlib.sha256(Path(config.decfg_path).read_bytes()).hexdigest(),
                      scope="Full33B noised-silence numerical probe; optimizer update discarded, not production training")
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
