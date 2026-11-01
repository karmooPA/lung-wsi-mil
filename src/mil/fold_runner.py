
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.data.mil_dataloader import create_train_val_dataloaders
from src.mil.checkpoint_manager import CheckpointManager
from src.mil.training import train_one_epoch


ModelFactory = Callable[[], nn.Module]
OptimizerFactory = Callable[[Any], Optimizer]
DataLoaderFactory = Callable[..., Any]
ValidationFunction = Callable[..., Mapping[str, Any]]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    if resolved.type == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available."
            )

    return resolved


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return (
            value.item()
            if value.numel() == 1
            else value.tolist()
        )

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {
            str(key): _to_python(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_python(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    raise TypeError(
        f"Unsupported result value type: {type(value).__name__}."
    )


def _finite_number(
    value: Any,
    name: str,
    allow_none: bool = False,
) -> float | None:
    value = _to_python(value)

    if value is None and allow_none:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be a numeric scalar."
        )

    if not math.isfinite(float(value)):
        raise ValueError(
            f"{name} must be finite."
        )

    return float(value)


def _validate_split_frame(
    frame: Any,
    name: str,
) -> None:
    if (
        frame is None
        or not hasattr(frame, "columns")
        or not hasattr(frame, "__len__")
    ):
        raise TypeError(
            f"{name} must be a pandas-like DataFrame."
        )

    if len(frame) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    required_columns = {
        "patient_id",
        "slide_id",
        "label",
    }

    missing_columns = required_columns.difference(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    columns = list(required_columns)

    if "case_id" in frame.columns:
        columns.append("case_id")

    if frame[columns].isna().any().any():
        raise ValueError(
            f"{name} contains missing identifiers or labels."
        )

    if frame["slide_id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate slide_id values."
        )

    for patient_id, slide_id, label in zip(
        frame["patient_id"].tolist(),
        frame["slide_id"].tolist(),
        frame["label"].tolist(),
    ):
        if (
            not isinstance(patient_id, str)
            or not patient_id.strip()
        ):
            raise ValueError(
                f"{name} contains an invalid patient_id."
            )

        if (
            not isinstance(slide_id, str)
            or not slide_id.strip()
        ):
            raise ValueError(
                f"{name} contains an invalid slide_id."
            )

        if (
            isinstance(label, (bool, np.bool_))
            or label not in (0, 1)
        ):
            raise ValueError(
                f"{name} contains non-binary labels."
            )

    if "case_id" in frame.columns:
        if not all(
            isinstance(case_id, str) and case_id.strip()
            for case_id in frame["case_id"].tolist()
        ):
            raise ValueError(
                f"{name} contains invalid case_id values."
            )


def _validate_fold_data(
    fold_data: Mapping[str, Any],
) -> tuple[int, Any, Any]:
    if not isinstance(fold_data, Mapping):
        raise TypeError(
            "fold_data must be a mapping."
        )

    required_keys = {
        "fold",
        "train",
        "val",
    }

    missing_keys = required_keys.difference(
        fold_data
    )

    if missing_keys:
        raise ValueError(
            f"fold_data is missing keys: {sorted(missing_keys)}."
        )

    fold = fold_data["fold"]

    if (
        isinstance(fold, bool)
        or not isinstance(fold, int)
        or not 1 <= fold <= 5
    ):
        raise ValueError(
            "Fold number must be an integer between 1 and 5."
        )

    train_frame = fold_data["train"]
    val_frame = fold_data["val"]

    _validate_split_frame(
        train_frame,
        "train split",
    )

    _validate_split_frame(
        val_frame,
        "validation split",
    )

    train_has_case = "case_id" in train_frame.columns
    val_has_case = "case_id" in val_frame.columns

    if train_has_case != val_has_case:
        raise ValueError(
            "Train and validation must use consistent case_id columns."
        )

    train_patients = set(
        train_frame["patient_id"].tolist()
    )

    val_patients = set(
        val_frame["patient_id"].tolist()
    )

    if train_patients.intersection(val_patients):
        raise ValueError(
            "Patient leakage detected between train and validation."
        )

    train_slides = set(
        train_frame["slide_id"].tolist()
    )

    val_slides = set(
        val_frame["slide_id"].tolist()
    )

    if train_slides.intersection(val_slides):
        raise ValueError(
            "Slide leakage detected between train and validation."
        )

    if train_has_case:
        train_cases = set(
            train_frame["case_id"].tolist()
        )

        val_cases = set(
            val_frame["case_id"].tolist()
        )

        if train_cases.intersection(val_cases):
            raise ValueError(
                "Case leakage detected between train and validation."
            )

    return fold, train_frame, val_frame


def _validate_train_result(
    result: Mapping[str, Any],
    expected_slides: int,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(
            "train_one_epoch must return a mapping."
        )

    required = {
        "loss",
        "num_batches",
        "num_bags",
        "num_slides",
    }

    missing = required.difference(result)

    if missing:
        raise ValueError(
            f"Training result is missing keys: {sorted(missing)}."
        )

    loss = _finite_number(
        result["loss"],
        "training loss",
    )

    if loss is None or loss < 0:
        raise ValueError(
            "Training loss must be non-negative."
        )

    counts = {}

    for key in (
        "num_batches",
        "num_bags",
        "num_slides",
    ):
        value = _to_python(result[key])

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{key} must be a non-negative integer."
            )

        counts[key] = value

    if counts["num_batches"] == 0:
        raise ValueError(
            "Training epoch produced no batches."
        )

    if counts["num_slides"] != expected_slides:
        raise ValueError(
            f"Training processed {counts['num_slides']} slides; "
            f"expected {expected_slides}."
        )

    expected_bags = 3 * expected_slides

    if counts["num_bags"] != expected_bags:
        raise ValueError(
            f"Training processed {counts['num_bags']} bags; "
            f"expected {expected_bags} for Original/Aug1/Aug2."
        )

    return {
        "loss": loss,
        **counts,
    }


def _validate_validation_result(
    result: Mapping[str, Any],
    expected_slide_labels: Mapping[str, int],
    threshold: float,
    expected_case_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(
            "validation_fn must return a mapping."
        )

    required = {
        "mean_loss",
        "num_evaluated",
        "num_correct",
        "probabilities",
        "true_labels",
        "predicted_labels",
        "slide_ids",
        "metrics",
    }

    missing = required.difference(result)

    if missing:
        raise ValueError(
            f"Validation result is missing keys: {sorted(missing)}."
        )

    mean_loss = _finite_number(
        result["mean_loss"],
        "validation mean_loss",
    )

    if mean_loss is None or mean_loss < 0:
        raise ValueError(
            "Validation loss must be non-negative."
        )

    num_evaluated = _to_python(
        result["num_evaluated"]
    )

    num_correct = _to_python(
        result["num_correct"]
    )

    if (
        isinstance(num_evaluated, bool)
        or not isinstance(num_evaluated, int)
        or num_evaluated <= 0
    ):
        raise ValueError(
            "num_evaluated must be a positive integer."
        )

    if (
        isinstance(num_correct, bool)
        or not isinstance(num_correct, int)
    ):
        raise TypeError(
            "num_correct must be an integer."
        )

    if not 0 <= num_correct <= num_evaluated:
        raise ValueError(
            "num_correct must be between zero and num_evaluated."
        )

    probabilities = _to_python(
        result["probabilities"]
    )

    true_labels = _to_python(
        result["true_labels"]
    )

    predicted_labels = _to_python(
        result["predicted_labels"]
    )

    slide_ids = _to_python(
        result["slide_ids"]
    )

    sequences = (
        probabilities,
        true_labels,
        predicted_labels,
        slide_ids,
    )

    if not all(
        isinstance(values, list)
        for values in sequences
    ):
        raise TypeError(
            "Validation outputs must be one-dimensional sequences."
        )

    if not (
        len(probabilities)
        == len(true_labels)
        == len(predicted_labels)
        == len(slide_ids)
        == num_evaluated
    ):
        raise ValueError(
            "Validation prediction, label, and slide_id lengths "
            "must equal num_evaluated."
        )

    if not all(
        isinstance(slide_id, str) and slide_id.strip()
        for slide_id in slide_ids
    ):
        raise TypeError(
            "Validation slide IDs must be non-empty strings."
        )

    if len(set(slide_ids)) != len(slide_ids):
        raise ValueError(
            "Duplicate validation slide IDs detected."
        )

    expected_slide_ids = set(
        expected_slide_labels
    )

    if set(slide_ids) != expected_slide_ids:
        missing_slides = expected_slide_ids.difference(
            slide_ids
        )

        extra_slides = set(slide_ids).difference(
            expected_slide_ids
        )

        raise ValueError(
            f"Validation coverage mismatch: "
            f"missing={len(missing_slides)}, "
            f"extra={len(extra_slides)}."
        )

    normalized_probabilities = []

    for probability in probabilities:
        value = _finite_number(
            probability,
            "validation probability",
        )

        if value is None or not 0.0 <= value <= 1.0:
            raise ValueError(
                "Validation probabilities must be finite "
                "and between 0 and 1."
            )

        normalized_probabilities.append(value)

    normalized_true_labels = []
    normalized_predicted_labels = []

    for label in true_labels:
        value = _to_python(label)

        if (
            isinstance(value, bool)
            or value not in (0, 1)
        ):
            raise ValueError(
                "true_labels must contain only binary values 0 and 1."
            )

        normalized_true_labels.append(
            int(value)
        )

    for label in predicted_labels:
        value = _to_python(label)

        if (
            isinstance(value, bool)
            or value not in (0, 1)
        ):
            raise ValueError(
                "predicted_labels must contain only binary values 0 and 1."
            )

        normalized_predicted_labels.append(
            int(value)
        )

    for slide_id, label in zip(
        slide_ids,
        normalized_true_labels,
    ):
        expected_label = expected_slide_labels[slide_id]

        if label != expected_label:
            raise ValueError(
                f"Ground-truth mismatch for slide {slide_id}: "
                f"received {label}, expected {expected_label}."
            )

    for probability, predicted_label in zip(
        normalized_probabilities,
        normalized_predicted_labels,
    ):
        expected_prediction = int(
            probability >= threshold
        )

        if predicted_label != expected_prediction:
            raise ValueError(
                "Predicted labels are inconsistent with "
                "the validation threshold."
            )

    calculated_correct = sum(
        predicted == actual
        for predicted, actual in zip(
            normalized_predicted_labels,
            normalized_true_labels,
        )
    )

    if calculated_correct != num_correct:
        raise ValueError(
            "num_correct does not match the validation predictions."
        )

    metrics = result["metrics"]

    if not isinstance(metrics, Mapping):
        raise TypeError(
            "Validation metrics must be a mapping."
        )

    required_metrics = {
        "accuracy",
        "auroc",
        "recall",
        "f1",
    }

    missing_metrics = required_metrics.difference(
        metrics
    )

    if missing_metrics:
        raise ValueError(
            f"Validation metrics are missing keys: "
            f"{sorted(missing_metrics)}."
        )

    normalized_metrics = {
        name: _finite_number(
            metrics[name],
            f"validation {name}",
            allow_none=(name == "auroc"),
        )
        for name in sorted(required_metrics)
    }

    for name, value in normalized_metrics.items():
        if value is None:
            continue

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Validation {name} must be between 0 and 1."
            )

    case_ids = _to_python(
        result.get("case_ids")
    )

    if expected_case_ids is not None:
        if (
            not isinstance(case_ids, list)
            or len(case_ids) != num_evaluated
        ):
            raise ValueError(
                "Validation must return one case_id per evaluated slide."
            )

    if case_ids is not None:
        if (
            not isinstance(case_ids, list)
            or len(case_ids) != num_evaluated
        ):
            raise ValueError(
                "case_ids must contain one identifier per evaluated slide."
            )

        if not all(
            isinstance(case_id, str) and case_id.strip()
            for case_id in case_ids
        ):
            raise ValueError(
                "Validation case_ids must be non-empty strings."
            )

        if expected_case_ids is not None:
            returned_case_ids = dict(
                zip(slide_ids, case_ids)
            )

            if returned_case_ids != dict(expected_case_ids):
                raise ValueError(
                    "Validation case_ids do not match the validation split."
                )

    diagnostics = _to_python(
        result.get("diagnostics", {})
    )

    if not isinstance(diagnostics, dict):
        raise TypeError(
            "Validation diagnostics must be a mapping."
        )

    metric_status = _to_python(
        result.get("metric_status")
    )

    if metric_status is None:
        metric_status = (
            "ok"
            if normalized_metrics["auroc"] is not None
            else "auroc_undefined"
        )

    if not isinstance(metric_status, str):
        raise TypeError(
            "metric_status must be a string."
        )

    normalized = {
        "mean_loss": mean_loss,
        "num_evaluated": num_evaluated,
        "num_correct": num_correct,
        "probabilities": normalized_probabilities,
        "true_labels": normalized_true_labels,
        "predicted_labels": normalized_predicted_labels,
        "slide_ids": slide_ids,
        "case_ids": case_ids,
        "metrics": normalized_metrics,
        "metric_status": metric_status,
        "diagnostics": diagnostics,
    }

    if "total_loss" in result:
        total_loss = _finite_number(
            result["total_loss"],
            "validation total_loss",
        )

        if total_loss is None or total_loss < 0:
            raise ValueError(
                "Validation total_loss must be non-negative."
            )

        normalized["total_loss"] = total_loss

    return normalized


def run_fold(
    fold_data: Mapping[str, Any],
    *,
    seed: int,
    dataset: Any,
    batch_size: int,
    model_factory: ModelFactory,
    optimizer_factory: OptimizerFactory,
    validation_fn: ValidationFunction,
    checkpoint_root: str | Path,
    epochs: int,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    eps: float = 1e-5,
    threshold: float = 0.5,
    config: Mapping[str, Any] | None = None,
    dataloader_factory: DataLoaderFactory = create_train_val_dataloaders,
) -> dict[str, Any]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError(
            "seed must be a non-negative integer."
        )

    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
    ):
        raise ValueError(
            "epochs must be a positive integer."
        )

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError(
            "batch_size must be a positive integer."
        )

    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError(
            "num_workers must be a non-negative integer."
        )

    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or not 0 < eps < 0.5
    ):
        raise ValueError(
            "eps must be finite and between 0 and 0.5."
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError(
            "threshold must be finite and between 0 and 1."
        )

    if not hasattr(dataset, "slide_ids"):
        raise TypeError(
            "dataset must expose slide_ids."
        )

    if not callable(model_factory):
        raise TypeError(
            "model_factory must be callable."
        )

    if not callable(optimizer_factory):
        raise TypeError(
            "optimizer_factory must be callable."
        )

    if not callable(dataloader_factory):
        raise TypeError(
            "dataloader_factory must be callable."
        )

    if not callable(validation_fn):
        raise TypeError(
            "validation_fn must be callable."
        )

    if config is not None and not isinstance(config, Mapping):
        raise TypeError(
            "config must be a mapping or None."
        )

    fold, train_frame, val_frame = _validate_fold_data(
        fold_data
    )

    dataset_slide_ids = list(dataset.slide_ids)

    if not all(
        isinstance(slide_id, str) and slide_id.strip()
        for slide_id in dataset_slide_ids
    ):
        raise ValueError(
            "Dataset slide_ids must be non-empty strings."
        )

    if len(set(dataset_slide_ids)) != len(dataset_slide_ids):
        raise ValueError(
            "Dataset contains duplicate slide IDs."
        )

    requested_slide_ids = (
        set(train_frame["slide_id"].tolist())
        | set(val_frame["slide_id"].tolist())
    )

    missing_slide_ids = requested_slide_ids.difference(
        dataset_slide_ids
    )

    if missing_slide_ids:
        raise ValueError(
            "Fold contains slides missing from the HDF5 dataset: "
            f"{sorted(missing_slide_ids)}."
        )

    resolved_device = _resolve_device(device)

    _set_seed(seed)

    model = model_factory()

    if not isinstance(model, nn.Module):
        raise TypeError(
            "model_factory must return a torch.nn.Module."
        )

    model = model.to(resolved_device)

    optimizer = optimizer_factory(
        model.parameters()
    )

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            "optimizer_factory must return a torch.optim.Optimizer."
        )

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    if (
        not optimizer_parameter_ids
        or not optimizer_parameter_ids.issubset(
            model_parameter_ids
        )
    ):
        raise ValueError(
            "Optimizer must reference parameters of this model."
        )

    loaders = dataloader_factory(
        dataset=dataset,
        fold_train_df=train_frame,
        fold_val_df=val_frame,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    if not isinstance(loaders, (tuple, list)) or len(loaders) != 2:
        raise ValueError(
            "dataloader_factory must return exactly "
            "(train_loader, val_loader)."
        )

    train_loader, val_loader = loaders

    if train_loader is None or val_loader is None:
        raise ValueError(
            "DataLoader factory returned None."
        )

    if train_loader is val_loader:
        raise ValueError(
            "Train and validation DataLoaders must be independent."
        )

    if (
        not hasattr(train_loader, "dataset")
        or not hasattr(val_loader, "dataset")
    ):
        raise TypeError(
            "Both DataLoaders must expose their datasets."
        )

    if len(train_loader.dataset) != len(train_frame):
        raise ValueError(
            "Training dataset size does not match the training split."
        )

    if len(val_loader.dataset) != len(val_frame):
        raise ValueError(
            "Validation dataset size does not match the validation split."
        )

    checkpoint_manager = CheckpointManager(
        checkpoint_root,
        seed,
        fold,
    )

    run_config = _to_python(
        dict(config or {})
    )

    if not isinstance(run_config, dict):
        raise TypeError(
            "config must be a mapping of JSON-compatible values."
        )

    run_config.update(
        {
            "seed": seed,
            "fold": fold,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(resolved_device),
            "eps": float(eps),
            "threshold": float(threshold),
            "training_views": [
                "Original",
                "Aug1",
                "Aug2",
            ],
            "validation_view": "Original",
            "early_stopping": False,
        }
    )

    expected_validation_labels = {
        str(slide_id): int(label)
        for slide_id, label in zip(
            val_frame["slide_id"].tolist(),
            val_frame["label"].tolist(),
        )
    }

    expected_case_ids = None

    if "case_id" in val_frame.columns:
        expected_case_ids = {
            str(slide_id): str(case_id)
            for slide_id, case_id in zip(
                val_frame["slide_id"].tolist(),
                val_frame["case_id"].tolist(),
            )
        }

    history = []
    final_train = None
    final_validation = None

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=resolved_device,
            eps=float(eps),
        )

        normalized_train = _validate_train_result(
            train_result,
            expected_slides=len(train_frame),
        )

        raw_validation = validation_fn(
            model,
            val_loader,
            device=resolved_device,
            threshold=float(threshold),
        )

        normalized_validation = _validate_validation_result(
            raw_validation,
            expected_slide_labels=expected_validation_labels,
            threshold=float(threshold),
            expected_case_ids=expected_case_ids,
        )

        epoch_record = {
            "epoch": epoch,
            "train": normalized_train,
            "validation": {
                "mean_loss": normalized_validation["mean_loss"],
                "num_evaluated": normalized_validation["num_evaluated"],
                "num_correct": normalized_validation["num_correct"],
                "metrics": normalized_validation["metrics"],
                "metric_status": normalized_validation["metric_status"],
                "diagnostics": normalized_validation["diagnostics"],
            },
        }

        history.append(epoch_record)

        final_train = normalized_train
        final_validation = normalized_validation

        checkpoint_manager.save(
            "last",
            model,
            optimizer,
            epoch=epoch,
            metric_name="validation.auroc",
            metric_mode="max",
            best_epoch=None,
            best_score=None,
            history={
                "epochs": history,
            },
            config=run_config,
            capture_rng=True,
        )

    return {
        "seed": seed,
        "fold": fold,
        "epochs_completed": len(history),
        "train_slides": len(train_frame),
        "validation_slides": len(val_frame),
        "final_train": final_train,
        "final_validation": final_validation,
        "history": history,
        "checkpoint_path": str(
            checkpoint_manager.path_for("last")
        ),
    }
