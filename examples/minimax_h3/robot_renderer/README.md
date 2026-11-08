# Native H3 robot renderer

One native training recipe and one inference path. This replaces the earlier
robot-specific silent-audio shell recipe; upstream general H3 examples are unchanged.
It uses the official Diffusers H3 model/reference packing, not the legacy robot
experiment loaders. No SolarRenderer control weights are loaded by this recipe.

## Configuration

`config.json` is shared by preparation, training and rollout:

| Setting | Default |
|---|---|
| Output clip | **158 total frames**, 24 fps |
| Training budget | **Continuous until a stop signal**; `steps` can set a finite budget |
| Adapter | Attention Q/K/V/output only; rank=alpha=32; zero output initialization |
| Frozen | H3 base, FFNs, text refiners, encoders, VAE |
| Optimizer | FP32 AdamW; LR5e-5, linear warmup100, then constant5e-5; betas0.9/0.95, eps1e-8, decay0.01 |
| Batch on32cards | SP8×DP4×accumulation2 =8 clips/update |
| Loss | Video-only data-ward flow matching; optional preservation weight0.1, default0 |
| DeCFG | Set `decfg_path` to the historical frozen Ref2VA adapter; active only in supervised training |
| Inverse-CFG / extra inference CFG | Disabled |
| Validation and checkpoint | At0/100/250 and every500updates;50 actual model evaluations |
| Precision | BF16 trunk; original FP32 projection/time owners; FP32 task adapter/optimizer/loss; FP64 positions |

The DCU loop uses separate FSDP and SP communicators over the same eight node ranks,
blockwise FSDP2 with managed FP32 adapters, one flat FP32 gradient
reduction across data replicas, explicit device synchronization and no activation recomputation, following
the earlier working launcher. Saved backward tensors are offloaded to CPU to fit the
158-frame native layout on64GB cards. This trades host memory and transfer time for VRAM.

Training consumes full158-frame windows in source order. Only the final remainder
is padded by repeating its final frame to the **smallest legal17n+5 length at or
above it**:140→141,150→158. No deliberately sampled short-frame buckets. Temporal
groups containing padding are omitted from the video loss. Existing low-rate
footage is resampled at shared physical timestamps;24fps does not invent observations.
For a5-frame target tail, only the free proxy reference is held to22frames to
satisfy Qwen's minimum video input; target length and supervised support stay unchanged.

Inference always generates158 frames. Consecutive windows share one boundary:
starts0,157,314,… . The previous generated last frame becomes the next native
image reference; its duplicate is discarded when joining. The last proxy window
is held to158, then generated output is trimmed to its actual support. Future
real frames are never used to continue the video.

## Environment and data

Use an H3-compatible Diffusers checkout containing `MiniMaxH3Transformer3DModel`
and its modular pipelines. The API used here corresponds to Diffusers revision
`90b4e34e79a86ec5e7f2437634fe95ecd2108796`; PyTorch2.10 was used for DCU checks.
Additional packages: `accelerate`, `safetensors`, `numpy`, `Pillow`, `av`.
The example is a namespace package; run the commands from the repository root.

The model directory has `transformer_ref`, `text_encoder`, `processor`, `vae`
and the official scheduler components. Model weights and datasets stay outside Git.
The conditioner keeps its first30 language blocks on GPU and the remainder on CPU.
The H3-local dtype query reads Linear tensor views directly during FSDP
weight gathering, avoiding an upstream Python3.10 fallback error.

Training manifest: JSONL, one real/proxy pair per row. Paths and captions are user
inputs; captions must be derived from task instructions/initial observations/proxy,
not future real-video frames. A description is wrapped in the six-section template;
an existing six-section prompt is retained.

```json
{"id":"droid_scene01_front","source":"droid","group":"scene01","split":"train","real":"/data/scene01/real.mp4","proxy":"/data/scene01/proxy.mp4","width":640,"height":352,"prompt":"A robot moves an object on a table."}
{"id":"droid_scene02_front","source":"droid","group":"scene02","split":"val","real":"/data/scene02/real.mp4","proxy":"/data/scene02/proxy.mp4","width":640,"height":352,"prompt":"A robot moves an object on a table."}
```

Keep all views/windows of a scene/episode in one split. `test` rows are excluded
from training and development validation. Previously inspected experiments remain
development evidence. The sampler selects source, episode-group, then window;
sources are equally likely. Preparation writes the lightweight `index.*.jsonl`
beside the tensor cache. All nodes must see the same cache/index and source paths,
either through shared storage or a replicated node-local cache.

## Prepare, train, roll out

```bash
python -m examples.minimax_h3.robot_renderer.prepare \
  --model /models/MiniMax-H3 --manifest /data/pairs.jsonl --out /data/h3-cache
```

Preparation can be split over independent GPU workers with `--shard N --shards K`.
Each worker writes its own index shard. It uses native high-resolution image
conditioning and the original free video-reference layout. Real targets and
proxies share geometry/time transforms and never pass through different lossy
transport encodings.

Training memoizes the **frozen native joint sampler's audio state before each of its50 steps**
when a clip/seed is first sampled. The first visit therefore includes a frozen teacher rollout.
This avoids an exhaustive two-seed teacher pass over every training window before launch.
For optional offline cache warming:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m examples.minimax_h3.robot_renderer.trajectories \
  --model /models/MiniMax-H3 --cache /data/h3-cache
```

Training samples a time index uniformly on that native50-step grid, reuses its
exact initial video noise, and pairs the noised real target with the corresponding
frozen native audio state. This replaces the old noised-silence input. It is
discrete-grid task flow matching, not audio supervision or four-step distillation.
Two trajectory seeds are available per clip; only requested trajectories are cached. The seed bank's coverage and generation
quality still need validation on the real robot data; the implementation alone
does not establish that earlier regression is solved.

The supplied controller launcher reuses the established container, Python and RCCL
settings. It first prepares native caches across32 workers, then replicates all training tensors for continuous runs (or only the exact scheduled
tensors for a finite budget), and submits all four training workers.
When DeCFG is configured, a full33B eight-card fixed-input probe must pass before
submission: repeatable backward, reversible DeCFG switching, frozen adapter integrity
and a discarded task-optimizer update. `--prepared` reuses completed preparation shards. Every node retains the complete index; sampling is still
from the full manifest. Use a fresh remote root for each immutable input/config snapshot.

```bash
python -m examples.minimax_h3.robot_renderer.launch_cluster \
  --nodes DCU3_direct DCU4_direct DCU5_direct DCU6_direct \
  --peers hygon3 hygon4 hygon5 hygon6 \
  --manifest /data/pairs.jsonl \
  --remote-root /data02/dex/h3_run_158f \
  --container-root /train/h3_run_158f --status /runs/h3_launch
```

The controller writes its current phase and exact commands to`launch.json`.
`preparing_native_158f_cache` is preparation; `training_submitted` records submitted
processes. Actual optimizer progress appears only in`run/train.jsonl`. Validation
uses two distinct groups per source, at three fixed seeds, including a step0 baseline.
Each validation call also generates158frames and trims unsupported tail output;
only training tails use shorter minimally padded legal lengths.

For manual submission, on each of four8-card nodes, set `NODE_RANK`, `MASTER_ADDR`, then run:

```bash
torchrun --nnodes=4 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29816 \
  -m examples.minimax_h3.robot_renderer.train \
  --model /models/MiniMax-H3 --cache /data/h3-cache --out /runs/robot-h3
```

Use `--config path.json` to change the shared configuration and
`--resume /runs/robot-h3/step_000100` to resume adapters, FP32 optimizer and RNGs.
Validation exports full clips and PSNR as a supporting score. Inspect robot
texture, motion, contacts and flicker alongside it; low supervised loss or PSNR
alone does not select the best model. There is no automatic image-quality stopping heuristic. With `steps: null`, training
continues until requested to stop. SIGTERM/SIGINT on a training worker requests a
coordinated stop after the current update and saves the final checkpoint. Checkpoints
include step0 and every validation update. The historical DeCFG learning-rate schedule is retained:100 updates of linear
warmup to5e-5, followed by constant5e-5 without decay.

Rollout manifest needs `id`, `reference` (initial image path), `proxy`, `width`,
`height`, and `prompt`; no real target video is required:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m examples.minimax_h3.robot_renderer.rollout \
  --model /models/MiniMax-H3 --manifest /data/rollouts.jsonl --out /runs/videos \
  --adapter /runs/robot-h3/step_001000/adapter.safetensors
```

Omit `--adapter` for the original-weight baseline. Generated-frame references can
drift over long rollouts, so complete-sequence results must be checked separately.

## Checks

```bash
python -m unittest discover -s examples/minimax_h3/robot_renderer/tests \
  -t . -p 'test_*.py' -v
torchrun --standalone --nproc_per_node=2 \
  -m examples.minimax_h3.robot_renderer.tests.check_parallel
```

The checks cover minimal tail padding, boundary coverage, masked tail loss,
native velocity sign, audio-state identity, adapter initialization/frozen base,
analytical attention gradients, uneven SP gradients and checkpoint recomputation.
On a real158-frame cache, full33B FSDP2 repeated backward also produced identical
losses and gradients. The old FSDP1 frozen/ignored-adapter wrapping did not qualify
for this native mixed-dtype model and is removed. The launcher/container/environment
are retained from the established DCU runs; this numerical repair does not establish
that adaptation improves video quality.

Check distributed updates, data-replica averaging and optimizer checkpoint roundtrip:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m examples.minimax_h3.robot_renderer.tests.check_training \
  --sequence-parallel 2 --out /tmp/h3-training-check
```

The same check runs across32 cards with SP8/DP4. It compares gradients, clipping and
three AdamW updates against a full unsharded reference, including a save/reload.
Reduced-model numerical checks do not establish full33B training quality.

## Training-only DeCFG

Set `decfg_path` in the launch configuration to the same `model_ref2va.safetensors`
used by the historical DeCFG run. The checkpoint has frozen rank64 branches over
the trunk and text-refiner attention/MLP projections. QKV output rows are deinterleaved
per head and SwiGLU output halves are swapped for the Diffusers layout. Scale is1;
no base weights are merged or repeatedly added/subtracted. Only the new rank32
attention task adapters are optimized and saved. DeCFG is off for baseline validation,
all sampled videos, frozen-native audio trajectories and deployment.

The repository default leaves the external checkpoint path unset; the cluster run
records its explicit path and SHA256 in the qualification artifact. Adding DeCFG
changes the training recipe and needs matched generation comparisons; it is not
proof of improved perceptual quality. Historical124-frame/20-step results cannot
be compared numerically to158-frame/50-step outputs without a shared evaluation protocol.

## Read-only training dashboard

Run `monitor.py` on the machine serving the results page. It uses existing SSH
aliases to read all four nodes once per minute, mirrors validation media with
rsync, and writes `index.html` and an atomic `snapshot.json` under `--out`.
It does not import the training model or modify remote training files/processes.
The local page reads the snapshot every 15 seconds, flags stale/disconnected data,
and supports checkpoint/source/seed filters and matched step0 comparisons.

```bash
python -m examples.minimax_h3.robot_renderer.monitor \
  --hosts node3 node4 node5 node6 \
  --remote-root /data/renderer-run --out /local/results/live
python -m http.server 8789 --bind 127.0.0.1 --directory /local/results
```

Open `http://127.0.0.1:8789/live/`. Use a range-capable HTTP server for video
seeking. `--once` collects one snapshot; otherwise keep the monitor running.
The browser shows Beijing time. Checkpoint0 is explicitly labelled untrained;
PSNR differences are supporting metrics, not a claim of perceptual improvement.
