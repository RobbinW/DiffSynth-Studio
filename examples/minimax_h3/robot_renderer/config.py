"""One configuration for cache preparation, training and inference."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    frames: int = 158
    fps: int = 24
    steps: int | None = None
    rank: int = 32
    learning_rate: float = 5e-5
    warmup: int = 100
    accumulation: int = 2
    sequence_parallel: int = 8
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    attention_backend: str = "scaled_sdpa"
    checkpoint_stride: int = 2
    inference_steps: int = 50
    seed: int = 42
    training_seeds: tuple = (42, 2026)
    validation_seeds: tuple = (42, 2026, 31415)
    validate_every: int = 500
    decfg_path: str | None = None

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def lr(self, update):
        return self.learning_rate * min(1.0, update / max(1, self.warmup))

    def validate_at(self, update):
        return update in (0, 100, 250, self.steps) or update % self.validate_every == 0
