# Native H3 robot renderer

One bounded training recipe and one inference path. This replaces the earlier
robot-specific silent-audio shell recipe; upstream general H3 examples are unchanged.
It uses the official Diffusers H3 model/reference packing, not the legacy robot
experiment loaders. No SolarRenderer control weights are loaded by this recipe.

## Configuration

`config.json` is shared by preparation, training and rollout:

| Setting | Default |
|---|---|
| Output clip | **158 total frames**, 24 fps |
| Training budget | **1,000 optimizer updates** |
| Adapter | Attention Q/K/V/output only; rank=alpha=32; zero output initialization |
| Frozen | H3 base, FFNs, text refiners, encoders, VAE |
| Optimizer | FP32 AdamW; LR1e-5→1e-6, warmup50, betas0.9/0.95, eps1e-8, decay0.01 |
| Batch on32cards | SP8×DP4×accumulation2 =8 clips/update |
| Loss | Video-only data-ward flow matching; optional preservation weight0.1, default0 |
| DeCFG / inverse-CFG / extra inference CFG | Disabled; no extra conditional branch |
| Validation | At0/25/50/100 and every100updates through1000;50 actual model evaluations |
| Precision | BF16 trunk; original FP32 projection/time owners; FP32 task adapter/optimizer/loss; FP64 positions |

Training consumes full158-frame windows in source order. Only the final remainder
is padded by repeating its final frame to the **smallest legal17n+5 length at or
above it**:140→141,150→158. No deliberately sampled short-frame buckets. Temporal
groups containing padding are omitted from the video loss. Existing low-rate
footage is resampled at shared physical timestamps;24fps does not invent observations.

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
The encoder offloads conditioner blocks to CPU between calls to limit GPU residency.

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

Cache the **frozen native joint sampler's audio state before each of its50 steps**:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m examples.minimax_h3.robot_renderer.trajectories \
  --model /models/MiniMax-H3 --cache /data/h3-cache
```

Training samples a time index uniformly on that native50-step grid, reuses its
exact initial video noise, and pairs the noised real target with the corresponding
frozen native audio state. This replaces the old noised-silence input. It is
discrete-grid task flow matching, not audio supervision or four-step distillation.
Two trajectory seeds are cached per clip. The seed bank's coverage and generation
quality still need validation on the real robot data; the implementation alone
does not establish that earlier regression is solved.

On each of four8-card nodes, set `NODE_RANK`, `MASTER_ADDR`, then run:

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
alone does not select the best model. There is no automatic quality heuristic or
unbounded continuation. Checkpoints include step0 and every validation update.

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
Reduced-model numerical checks do not establish full33B training quality.
