"""One configuration for cache preparation, training and inference."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    frames: int = 158
    fps: int = 24
    steps: int | None = None
    rank: int = 32
    learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-6
    warmup: int = 50
    lr_decay_steps: int = 1000
    accumulation: int = 2
    sequence_parallel: int = 8
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    preserve_weight: float = 0.0
    inference_steps: int = 50
    seed: int = 42
    trajectory_seeds: tuple = (42, 2026)
    validation_seeds: tuple = (42, 2026, 31415)
    validate_every: int = 500
    decfg_path: str | None = None

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def lr(self, update):
        if update <= self.warmup:
            return self.learning_rate * update / self.warmup
        progress = min(1.0, (update - self.warmup) / (self.lr_decay_steps - self.warmup))
        return self.minimum_learning_rate + 0.5 * (
            self.learning_rate - self.minimum_learning_rate
        ) * (1 + math.cos(math.pi * progress))

    def validate_at(self, update):
        return update in (0, 100, 250, self.steps) or update % self.validate_every == 0
