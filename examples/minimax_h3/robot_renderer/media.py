"""Shared timestamp resampling; no intermediate lossy transport videos."""

import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def read_manifest(path):
    path = Path(path)
    rows = json.loads(path.read_text()) if path.suffix == ".json" else [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    return rows


def read_index(cache):
    return [row for path in sorted(Path(cache).glob("index.*.jsonl")) for row in read_manifest(path)]


def read_video(path, width, height):
    images, timestamps = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            timestamps.append(float(frame.time))
            images.append(frame.to_image().convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS))
    return images, np.asarray(timestamps)


def resample(images, timestamps, times):
    right = np.searchsorted(timestamps, times).clip(0, len(images) - 1)
    left = (right - 1).clip(0)
    indices = np.where(abs(timestamps[left] - times) < abs(timestamps[right] - times), left, right)
    return [images[index] for index in indices]


def read_pair(row, fps=24):
    real, real_pts = read_video(row["real"], row["width"], row["height"])
    proxy, proxy_pts = read_video(row["proxy"], row["width"], row["height"])
    # Pixel timestamps, rather than an approximate container duration, define support.
    end = min(real_pts[-1], proxy_pts[-1])
    times = np.arange(int(np.floor(end * fps + 1e-6)) + 1) / fps
    return resample(real, real_pts, times), resample(proxy, proxy_pts, times)


def write_video(frames, path, fps=24):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def structured_prompt(description):
    if description.startswith("subject_definitions:"):
        return description
    return (
        "subject_definitions:\n<Image 1> defines scene appearance. "
        "<Video 1> defines robot/object motion and the camera.\n\n"
        f"summary:\n{description}\n\n"
        "retention_analysis:\nPreserve the real image's robot identity, materials, "
        "textures, lighting and background. Use the proxy for geometry and timing.\n\n"
        f"detailed_description:\n{description}\n"
        "Follow the proxy throughout the clip.\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )
