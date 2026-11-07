"""Physical-time windows, minimal tail padding and shared rollout boundaries."""

from dataclasses import dataclass


def legal_frames(count):
    """Smallest 17n+5 length covering count; padding never drops real frames."""
    if count < 1:
        raise ValueError("A window must contain at least one frame")
    return 5 + 17 * max(0, (count - 5 + 16) // 17)


def latent_frames(count):
    if count < 5 or (count - 5) % 17:
        raise ValueError("H3 frame counts must be 17n+5")
    return 5 * ((count - 5) // 17) + 2


def valid_latents(valid_frames, padded_frames):
    """Exclude temporal groups containing repeated tail pixels from the loss."""
    end = 0
    mask = []
    for index in range(latent_frames(padded_frames)):
        end += (1, 4, 4, 4, 4)[index % 5]
        mask.append(end <= valid_frames)
    return mask


@dataclass(frozen=True)
class Window:
    start: int
    valid: int
    frames: int


def training_windows(total, frames=158):
    """Use full windows followed by one minimally padded tail; no short buckets."""
    if frames != legal_frames(frames):
        raise ValueError("Window size must be 17n+5")
    return [Window(start, min(frames, total - start), legal_frames(min(frames, total - start)))
            for start in range(0, total, frames)]


def rollout_windows(total, frames=158):
    """Each call generates frames, sharing one boundary with the previous call."""
    if frames <= 1 or frames != legal_frames(frames):
        raise ValueError("Rollout size must be a legal multi-frame window")
    windows = []
    start = 0
    while start < total:
        valid = min(frames, total - start)
        windows.append(Window(start, valid, frames))
        if start + valid >= total:
            break
        start += frames - 1
    return windows


def pad_tail(frames, length):
    return list(frames) + [frames[-1]] * (length - len(frames))
