"""Encode missing-audio silence once per duration; no online teacher model."""

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKLMiniMaxH3Audio


@torch.inference_mode()
def encode_silence(model_path, frames, fps=24, device="cuda"):
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        model_path, subfolder="audio_vae", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, 1, -1)
    std = torch.tensor(vae.config.latents_std).view(1, 1, -1)
    result = {}
    for count in sorted(set(frames)):
        # Match the official silent_on_missing_audio waveform: stereo 32kHz,
        # rounded to the H3 audio VAE's 800-sample hop (40 latent frames/s).
        length = round(count / fps * 40)
        waveform = torch.zeros(2, 1, 800 * length, device=device, dtype=vae.dtype)
        posterior = vae.encode(waveform, return_dict=False)[0]
        rows = (posterior.mode().float().cpu().transpose(1, 2) - mean) / std
        rows = rows.reshape(-1, vae.config.latent_channels).contiguous()
        assert rows.shape == (2 * length, 32) and torch.isfinite(rows).all()
        result[count] = rows
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    values = encode_silence(args.model, args.frames, args.fps)
    path = Path(args.out)
    temporary = path.with_suffix(".tmp")
    torch.save(values, temporary)
    temporary.replace(path)
    print({frames: list(rows.shape) for frames, rows in values.items()}, flush=True)


if __name__ == "__main__":
    main()
