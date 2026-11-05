#!/usr/bin/env bash
set -euo pipefail

# Supply aligned real/proxy clips and a real reference frame in metadata.json.
# Keep input_audio pointing to the real video so missing soundtracks use the
# official silence fallback. Only the video loss contributes to optimization.
# Prepare 640x352 or 512x384 clips before invoking this mixed-resolution recipe.
: "${DATASET_BASE_PATH:?Set DATASET_BASE_PATH to the prepared dataset directory}"
: "${DATASET_METADATA_PATH:?Set DATASET_METADATA_PATH to metadata.json}"
CACHE_PATH="${CACHE_PATH:-./models/train/robot-ref2va-silent-cache}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/robot-ref2va-video-loss-lora}"

common_args=(
  --data_file_keys "video,input_audio,references"
  --extra_inputs "input_audio,references"
  --silent_on_missing_audio
  --audio_loss_weight 0
  --max_pixels 225280
  --num_frames 124
  --learning_rate 1e-4
  --remove_prefix_in_ckpt "pipe.dit."
  --lora_base_model "dit"
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
  --lora_rank 32
  --use_gradient_checkpointing
)

# DeCFG is optional upstream. This recipe keeps training_cfg_scale=1 and
# records a plain-flow-matching baseline before a separate DeCFG comparison.
accelerate launch examples/minimax_h3/model_training/train.py \
  "${common_args[@]}" \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --dataset_repeat 1 \
  --num_epochs 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --output_path "$CACHE_PATH" \
  --task "sft:data_process"

accelerate launch examples/minimax_h3/model_training/train.py \
  "${common_args[@]}" \
  --dataset_base_path "$CACHE_PATH" \
  --dataset_repeat 100 \
  --num_epochs 5 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --output_path "$OUTPUT_PATH" \
  --save_steps 500 \
  --find_unused_parameters \
  --task "sft:train"
