
from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


class CheckpointManager:
    def __init__(self, root_dir: str | Path, seed: int, fold: int) -> None:
        if not isinstance(root_dir, (str, Path)) or not str(root_dir).strip():
            raise ValueError("root_dir must be a non-empty path.")

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        if isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= 5:
            raise ValueError("fold must be an integer between 1 and 5.")

        self.seed = seed
        self.fold = fold
        self.directory = (
            Path(root_dir).expanduser()
            / f"seed_{seed}"
            / f"fold_{fold}"
        )

    def path_for(self, kind: str) -> Path:
        if kind not in ("last", "best"):
            raise ValueError("kind must be 'last' or 'best'.")

        return self.directory / f"{kind}.pt"

    def exists(self, kind: str) -> bool:
        return self.path_for(kind).is_file()

    @staticmethod
    def _mps_available() -> bool:
        backend = getattr(
            getattr(torch, "backends", None),
            "mps",
            None,
        )
        mps_api = getattr(torch, "mps", None)

        return bool(
            backend is not None
            and callable(getattr(backend, "is_available", None))
            and backend.is_available()
            and mps_api is not None
            and callable(getattr(mps_api, "get_rng_state", None))
            and callable(getattr(mps_api, "set_rng_state", None))
        )

    @staticmethod
    def _check_optimizer_model(
        model: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        model_parameters = {
            id(parameter)
            for parameter in model.parameters()
        }

        optimizer_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        if (
            not optimizer_parameters
            or not optimizer_parameters.issubset(model_parameters)
        ):
            raise ValueError(
                "Optimizer must reference parameters of this model."
            )

    @staticmethod
    def _json_metadata(
        value: Mapping[str, Any] | None,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}

        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")

        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings.")

        try:
            return json.loads(
                json.dumps(
                    dict(value),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain JSON-compatible finite values."
            ) from exc

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        np_state = np.random.get_state()

        mps_state = None

        if CheckpointManager._mps_available():
            mps_state = torch.mps.get_rng_state()

        return {
            "python": random.getstate(),
            "numpy": (
                str(np_state[0]),
                np_state[1].tolist(),
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            ),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "mps": mps_state,
        }

    @staticmethod
    def _restore_rng(
        state: Mapping[str, Any],
        require_mps: bool = False,
    ) -> None:
        if not isinstance(state, Mapping):
            raise ValueError(
                "Checkpoint does not contain a valid RNG state."
            )

        required = {
            "python",
            "numpy",
            "torch",
            "cuda",
        }

        if not required.issubset(state):
            raise ValueError(
                "Checkpoint is missing required RNG state fields."
            )

        cuda_state = state["cuda"]
        mps_state = state.get("mps")

        if cuda_state is not None:
            if not isinstance(cuda_state, (list, tuple)):
                raise ValueError("Invalid CUDA RNG state.")

            if (
                not torch.cuda.is_available()
                or len(cuda_state) != torch.cuda.device_count()
            ):
                raise RuntimeError(
                    "Cannot restore CUDA RNG state on this device setup."
                )

        if mps_state is not None:
            if not CheckpointManager._mps_available():
                raise RuntimeError(
                    "Cannot restore MPS RNG state because MPS is unavailable."
                )

            if not isinstance(mps_state, torch.Tensor):
                raise ValueError("Invalid MPS RNG state.")

        elif require_mps:
            raise RuntimeError(
                "Checkpoint does not contain MPS RNG state. "
                "Exact MPS RNG restoration is not possible."
            )

        random.setstate(state["python"])

        np_state = state["numpy"]

        np.random.set_state(
            (
                np_state[0],
                np.asarray(
                    np_state[1],
                    dtype=np.uint32,
                ),
                np_state[2],
                np_state[3],
                np_state[4],
            )
        )

        torch.set_rng_state(
            state["torch"].cpu()
        )

        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        if mps_state is not None:
            torch.mps.set_rng_state(
                mps_state.cpu()
            )

    def save(
        self,
        kind: str,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        epoch: int,
        metric_name: str,
        metric_mode: str,
        best_epoch: int | None,
        best_score: float | None,
        history: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        capture_rng: bool = True,
    ) -> Path:
        destination = self.path_for(kind)

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")

        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer."
            )

        self._check_optimizer_model(
            model,
            optimizer,
        )

        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 1
        ):
            raise ValueError(
                "epoch must be a positive integer."
            )

        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric_name must be non-empty.")

        if metric_mode not in ("min", "max"):
            raise ValueError("metric_mode must be 'min' or 'max'.")

        if (best_epoch is None) != (best_score is None):
            raise ValueError(
                "best_epoch and best_score must be set together."
            )

        if best_epoch is not None:
            if (
                isinstance(best_epoch, bool)
                or not isinstance(best_epoch, int)
                or not 1 <= best_epoch <= epoch
            ):
                raise ValueError(
                    "best_epoch must be between 1 and epoch."
                )

            if (
                isinstance(best_score, bool)
                or not isinstance(best_score, (int, float))
                or not math.isfinite(best_score)
            ):
                raise ValueError(
                    "best_score must be a finite number."
                )

        if kind == "best" and (
            best_epoch != epoch or best_score is None
        ):
            raise ValueError(
                "A best checkpoint must refer to its own epoch and score."
            )

        if not isinstance(capture_rng, bool):
            raise TypeError(
                "capture_rng must be a boolean."
            )

        payload = {
            "format_version": 1,
            "kind": kind,
            "seed": self.seed,
            "fold": self.fold,
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_mode": metric_mode,
            "best_epoch": best_epoch,
            "best_score": (
                float(best_score)
                if best_score is not None
                else None
            ),
            "history": self._json_metadata(
                history,
                "history",
            ),
            "config": self._json_metadata(
                config,
                "config",
            ),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": (
                self._rng_state()
                if capture_rng
                else None
            ),
        }

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{kind}_",
            suffix=".tmp",
            dir=self.directory,
        )

        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(
                    payload,
                    stream,
                )

                stream.flush()
                os.fsync(stream.fileno())

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

        return destination

    def load(
        self,
        kind: str,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        *,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        source = self.path_for(kind)

        if not source.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {source}"
            )

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be a torch.nn.Module."
            )

        if (
            optimizer is not None
            and not isinstance(optimizer, Optimizer)
        ):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer or None."
            )

        if optimizer is not None:
            self._check_optimizer_model(
                model,
                optimizer,
            )

        checkpoint = torch.load(
            source,
            map_location=map_location,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Checkpoint must contain a dictionary."
            )

        required = {
            "format_version",
            "kind",
            "seed",
            "fold",
            "epoch",
            "metric_name",
            "metric_mode",
            "best_epoch",
            "best_score",
            "history",
            "config",
            "model_state",
            "optimizer_state",
            "rng_state",
        }

        if not required.issubset(checkpoint):
            raise ValueError(
                "Checkpoint is missing required fields."
            )

        if checkpoint["format_version"] != 1:
            raise ValueError(
                "Unsupported checkpoint format version."
            )

        if (
            checkpoint["seed"] != self.seed
            or checkpoint["fold"] != self.fold
            or checkpoint["kind"] != kind
        ):
            raise ValueError(
                "Checkpoint does not belong to this seed, fold, or kind."
            )

        if restore_rng and checkpoint["rng_state"] is None:
            raise ValueError(
                "Checkpoint was saved without RNG state."
            )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        if restore_rng:
            model_uses_mps = any(
                parameter.device.type == "mps"
                for parameter in model.parameters()
            )

            self._restore_rng(
                checkpoint["rng_state"],
                require_mps=model_uses_mps,
            )

        return {
            key: checkpoint[key]
            for key in (
                "kind",
                "seed",
                "fold",
                "epoch",
                "metric_name",
                "metric_mode",
                "best_epoch",
                "best_score",
                "history",
                "config",
            )
        }
