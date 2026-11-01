"""Explicit engineering defaults; unspecified paper hyperparameters are not inferred."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (2, 3, 5, 7, 11)
    folds: int = 5
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0001
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    batch_size: int = 2
    num_workers: int = 0
    num_threads: int = 2
    attention_dim: int = 128
    threshold: float = 0.5
    device: str = "cpu"
    bootstrap_resamples: int = 1000
    heatmap_limit: int = 6
    heatmap_max_size: int = 1024
    allow_unverified_features: bool = False

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
                for seed in self.seeds
            )
        ):
            raise ValueError("seeds must contain distinct nonnegative 32-bit integers.")
        for name in (
            "folds",
            "epochs",
            "patience",
            "batch_size",
            "num_threads",
            "attention_dim",
            "heatmap_max_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.folds != 5:
            raise ValueError("The saved reference protocol requires exactly five folds.")
        if self.heatmap_max_size > 4096:
            raise ValueError("heatmap_max_size must not exceed 4096 pixels.")
        for name in ("num_workers", "bootstrap_resamples", "heatmap_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("learning_rate", "weight_decay", "min_delta", "threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError(
                "learning_rate must be positive; weight_decay and min_delta nonnegative."
            )
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0,1].")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty torch device string.")
        if not isinstance(self.allow_unverified_features, bool):
            raise ValueError("allow_unverified_features must be a boolean.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values).difference(field.name for field in fields(cls))
        if unknown:
            raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
