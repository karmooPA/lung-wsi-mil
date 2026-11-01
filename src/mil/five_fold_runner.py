
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.data.mil_dataloader import create_train_val_dataloaders
from src.mil.fold_runner import run_fold


def _scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value

    return value


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
        f"Unsupported value type: {type(value).__name__}."
    )


def _json_compatible(
    value: Any,
    name: str,
) -> Any:
    converted = _to_python(value)

    try:
        return json.loads(
            json.dumps(
                converted,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain JSON-compatible finite values."
        ) from exc


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


def _records(
    frame: Any,
    name: str,
) -> tuple[dict[str, tuple[str, int, str | None]], bool]:
    if (
        frame is None
        or not hasattr(frame, "columns")
        or not hasattr(frame, "__len__")
    ):
        raise TypeError(
            f"{name} must be a pandas-like DataFrame."
        )

    required = {
        "patient_id",
        "slide_id",
        "label",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}."
        )

    if len(frame) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    has_case_id = "case_id" in frame.columns

    columns = [
        "slide_id",
        "patient_id",
        "label",
    ]

    if has_case_id:
        columns.append("case_id")

    if frame[columns].isna().any().any():
        raise ValueError(
            f"{name} contains missing identifiers or labels."
        )

    if frame["slide_id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate slide_id values."
        )

    result = {}

    for row in frame[columns].itertuples(
        index=False,
        name=None,
    ):
        slide_id = _scalar(row[0])
        patient_id = _scalar(row[1])
        label = _scalar(row[2])
        case_id = (
            _scalar(row[3])
            if has_case_id
            else None
        )

        if (
            not isinstance(slide_id, str)
            or not slide_id.strip()
        ):
            raise ValueError(
                f"{name} contains an invalid slide_id."
            )

        if (
            not isinstance(patient_id, str)
            or not patient_id.strip()
        ):
            raise ValueError(
                f"{name} contains an invalid patient_id."
            )

        if (
            isinstance(label, (bool, np.bool_))
            or label not in (0, 1)
        ):
            raise ValueError(
                f"{name} labels must contain only 0 and 1."
            )

        if has_case_id and (
            not isinstance(case_id, str)
            or not case_id.strip()
        ):
            raise ValueError(
                f"{name} contains an invalid case_id."
            )

        result[slide_id] = (
            patient_id,
            int(label),
            case_id,
        )

    return result, has_case_id


def _validate_folds(
    folds: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    set[str],
    dict[str, tuple[str, int, str | None]],
    dict[str, int],
]:
    if (
        isinstance(folds, (str, bytes))
        or not isinstance(folds, Sequence)
    ):
        raise TypeError(
            "folds must be a sequence of five fold mappings."
        )

    if len(folds) != 5:
        raise ValueError(
            f"Exactly five folds are required; received {len(folds)}."
        )

    by_number = {}

    for fold_data in folds:
        if not isinstance(fold_data, Mapping):
            raise TypeError(
                "Every fold must be a mapping."
            )

        number = fold_data.get("fold")

        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number not in range(1, 6)
        ):
            raise ValueError(
                "Fold numbers must be integers from 1 through 5."
            )

        if number in by_number:
            raise ValueError(
                f"Fold {number} appears more than once."
            )

        if (
            "train" not in fold_data
            or "val" not in fold_data
        ):
            raise ValueError(
                f"Fold {number} must contain train and val."
            )

        by_number[number] = fold_data

    if set(by_number) != {1, 2, 3, 4, 5}:
        raise ValueError(
            "Fold numbers must be exactly 1, 2, 3, 4, and 5."
        )

    ordered = [
        by_number[number]
        for number in range(1, 6)
    ]

    outer_pool = None
    slide_metadata = {}
    patient_slides: dict[str, set[str]] = {}
    case_to_patient = {}
    validation_slides = {}
    validation_patients = {}
    validation_cases = {}
    has_case_id = None

    for fold_data in ordered:
        number = fold_data["fold"]

        train, train_has_case = _records(
            fold_data["train"],
            f"fold {number} train",
        )

        val, val_has_case = _records(
            fold_data["val"],
            f"fold {number} validation",
        )

        if train_has_case != val_has_case:
            raise ValueError(
                f"Fold {number} has inconsistent case_id columns."
            )

        if has_case_id is None:
            has_case_id = train_has_case
        elif has_case_id != train_has_case:
            raise ValueError(
                "case_id must be present consistently across all folds."
            )

        train_slides = set(train)
        val_slides = set(val)

        train_patients = {
            item[0]
            for item in train.values()
        }

        val_patients = {
            item[0]
            for item in val.values()
        }

        if train_slides.intersection(val_slides):
            raise ValueError(
                f"Slide leakage detected in fold {number}."
            )

        if train_patients.intersection(val_patients):
            raise ValueError(
                f"Patient leakage detected in fold {number}."
            )

        if has_case_id:
            train_cases = {
                item[2]
                for item in train.values()
            }

            val_cases = {
                item[2]
                for item in val.values()
            }

            if train_cases.intersection(val_cases):
                raise ValueError(
                    f"Case leakage detected in fold {number}."
                )

        current_pool = train_slides.union(val_slides)

        if outer_pool is None:
            outer_pool = current_pool
        elif current_pool != outer_pool:
            raise ValueError(
                f"Fold {number} does not use the same outer training pool."
            )

        for slide_id, metadata in {
            **train,
            **val,
        }.items():
            previous = slide_metadata.get(slide_id)

            if previous is not None and previous != metadata:
                raise ValueError(
                    f"Conflicting metadata for slide {slide_id}."
                )

            slide_metadata[slide_id] = metadata

            patient_id, _, case_id = metadata

            patient_slides.setdefault(
                patient_id,
                set(),
            ).add(slide_id)

            if case_id is not None:
                previous_patient = case_to_patient.get(case_id)

                if (
                    previous_patient is not None
                    and previous_patient != patient_id
                ):
                    raise ValueError(
                        f"Case {case_id} is associated with multiple patients."
                    )

                case_to_patient[case_id] = patient_id

        for slide_id in val_slides:
            if slide_id in validation_slides:
                raise ValueError(
                    f"Slide {slide_id} appears in multiple validation folds."
                )

            validation_slides[slide_id] = number

        for patient_id in val_patients:
            if patient_id in validation_patients:
                raise ValueError(
                    f"Patient {patient_id} appears in multiple validation folds."
                )

            validation_patients[patient_id] = number

        if has_case_id:
            for case_id in {
                item[2]
                for item in val.values()
            }:
                if case_id in validation_cases:
                    raise ValueError(
                        f"Case {case_id} appears in multiple validation folds."
                    )

                validation_cases[case_id] = number

        if train_slides != outer_pool.difference(val_slides):
            raise ValueError(
                f"Fold {number} training slides are not the complement "
                "of its validation slides."
            )

    if not outer_pool:
        raise ValueError(
            "The outer training pool is empty."
        )

    if set(validation_slides) != outer_pool:
        raise ValueError(
            "Every outer-training slide must appear in validation exactly once."
        )

    if set(validation_patients) != set(patient_slides):
        raise ValueError(
            "Every outer-training patient must appear in validation exactly once."
        )

    for patient_id, slide_ids in patient_slides.items():
        assigned_folds = {
            validation_slides.get(slide_id)
            for slide_id in slide_ids
        }

        if None in assigned_folds or len(assigned_folds) != 1:
            raise ValueError(
                f"All slides of patient {patient_id} must belong "
                "to one validation fold."
            )

    if has_case_id and set(validation_cases) != set(case_to_patient):
        raise ValueError(
            "Every outer-training case must appear in validation exactly once."
        )

    return (
        ordered,
        outer_pool,
        slide_metadata,
        validation_slides,
    )


def _unique_factory(
    factory: Callable[..., Any],
    references: list[weakref.ReferenceType[Any]],
    name: str,
) -> Callable[..., Any]:
    if not callable(factory):
        raise TypeError(
            f"{name} must be callable."
        )

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        instance = factory(*args, **kwargs)

        references[:] = [
            reference
            for reference in references
            if reference() is not None
        ]

        if any(
            reference() is instance
            for reference in references
        ):
            raise ValueError(
                f"{name} must create a fresh object for every fold."
            )

        try:
            references.append(
                weakref.ref(instance)
            )
        except TypeError as exc:
            raise TypeError(
                f"Objects returned by {name} must support weak references."
            ) from exc

        return instance

    return wrapped


def _aggregate(
    values: list[float | None],
) -> dict[str, float | int | None]:
    defined = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "Metric aggregation requires numeric values or None."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                "Metric aggregation does not accept NaN or Inf."
            )

        defined.append(
            float(value)
        )

    if not defined:
        return {
            "mean": None,
            "std": None,
            "n_defined": 0,
        }

    return {
        "mean": statistics.fmean(defined),
        "std": (
            statistics.stdev(defined)
            if len(defined) > 1
            else None
        ),
        "n_defined": len(defined),
    }


def _write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}_",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            stream.write("\n")
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


def run_five_folds(
    folds: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    dataset: Any,
    batch_size: int,
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[Any], Optimizer],
    validation_fn: Callable[..., Mapping[str, Any]],
    checkpoint_root: str | Path,
    epochs: int,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    eps: float = 1e-5,
    threshold: float = 0.5,
    config: Mapping[str, Any] | None = None,
    results_path: str | Path | None = None,
    dataloader_factory: Callable[..., Any] = create_train_val_dataloaders,
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

    if not callable(model_factory):
        raise TypeError(
            "model_factory must be callable."
        )

    if not callable(optimizer_factory):
        raise TypeError(
            "optimizer_factory must be callable."
        )

    if not callable(validation_fn):
        raise TypeError(
            "validation_fn must be callable."
        )

    if not callable(dataloader_factory):
        raise TypeError(
            "dataloader_factory must be callable."
        )

    if not hasattr(dataset, "slide_ids"):
        raise TypeError(
            "dataset must expose slide_ids."
        )

    if config is not None and not isinstance(config, Mapping):
        raise TypeError(
            "config must be a mapping or None."
        )

    resolved_device = torch.device(device)

    (
        ordered_folds,
        outer_pool,
        slide_metadata,
        validation_slides,
    ) = _validate_folds(folds)

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

    missing_dataset_slides = outer_pool.difference(
        dataset_slide_ids
    )

    if missing_dataset_slides:
        raise ValueError(
            "Outer training pool contains slides missing from the dataset: "
            f"{sorted(missing_dataset_slides)}."
        )

    config_payload = _json_compatible(
        dict(config or {}),
        "config",
    )

    if not isinstance(config_payload, dict):
        raise TypeError(
            "config must resolve to a dictionary."
        )

    model_refs: list[weakref.ReferenceType[Any]] = []
    optimizer_refs: list[weakref.ReferenceType[Any]] = []

    fresh_model_factory = _unique_factory(
        model_factory,
        model_refs,
        "model_factory",
    )

    fresh_optimizer_factory = _unique_factory(
        optimizer_factory,
        optimizer_refs,
        "optimizer_factory",
    )

    fold_results = []
    seen_validation_slide_ids = set()

    for fold_data in ordered_folds:
        result = run_fold(
            fold_data,
            seed=seed,
            dataset=dataset,
            batch_size=batch_size,
            model_factory=fresh_model_factory,
            optimizer_factory=fresh_optimizer_factory,
            validation_fn=validation_fn,
            checkpoint_root=checkpoint_root,
            epochs=epochs,
            num_workers=num_workers,
            device=resolved_device,
            eps=eps,
            threshold=threshold,
            config=config_payload,
            dataloader_factory=dataloader_factory,
        )

        if not isinstance(result, Mapping):
            raise TypeError(
                "run_fold must return a mapping."
            )

        required_result_keys = {
            "seed",
            "fold",
            "epochs_completed",
            "train_slides",
            "validation_slides",
            "final_train",
            "final_validation",
            "history",
            "checkpoint_path",
        }

        missing_result_keys = required_result_keys.difference(
            result
        )

        if missing_result_keys:
            raise ValueError(
                f"Fold result is missing keys: "
                f"{sorted(missing_result_keys)}."
            )

        if (
            result["seed"] != seed
            or result["fold"] != fold_data["fold"]
        ):
            raise RuntimeError(
                "run_fold returned mismatched seed or fold metadata."
            )

        if result["epochs_completed"] != epochs:
            raise RuntimeError(
                f"Fold {fold_data['fold']} completed "
                f"{result['epochs_completed']} epochs; expected {epochs}."
            )

        expected_fold_slides = {
            slide_id
            for slide_id, assigned_fold in validation_slides.items()
            if assigned_fold == fold_data["fold"]
        }

        expected_train_count = (
            len(outer_pool)
            - len(expected_fold_slides)
        )

        if result["train_slides"] != expected_train_count:
            raise RuntimeError(
                f"Fold {fold_data['fold']} returned an incorrect "
                "training slide count."
            )

        if result["validation_slides"] != len(expected_fold_slides):
            raise RuntimeError(
                f"Fold {fold_data['fold']} returned an incorrect "
                "validation slide count."
            )

        final_train = result["final_train"]

        if not isinstance(final_train, Mapping) or "loss" not in final_train:
            raise ValueError(
                "Fold result contains an invalid final_train."
            )

        final_train_loss = _finite_number(
            final_train["loss"],
            "final training loss",
        )

        if final_train_loss is None or final_train_loss < 0:
            raise ValueError(
                "Final training loss must be non-negative."
            )

        validation = result["final_validation"]

        if not isinstance(validation, Mapping):
            raise TypeError(
                "final_validation must be a mapping."
            )

        required_validation_keys = {
            "mean_loss",
            "num_evaluated",
            "num_correct",
            "probabilities",
            "true_labels",
            "predicted_labels",
            "slide_ids",
            "metrics",
        }

        missing_validation_keys = required_validation_keys.difference(
            validation
        )

        if missing_validation_keys:
            raise ValueError(
                "Final validation result is missing keys: "
                f"{sorted(missing_validation_keys)}."
            )

        slide_ids = _to_python(
            validation["slide_ids"]
        )

        true_labels = _to_python(
            validation["true_labels"]
        )

        predicted_labels = _to_python(
            validation["predicted_labels"]
        )

        probabilities = _to_python(
            validation["probabilities"]
        )

        sequences = (
            slide_ids,
            true_labels,
            predicted_labels,
            probabilities,
        )

        if not all(
            isinstance(values, list)
            for values in sequences
        ):
            raise TypeError(
                "Final validation outputs must be one-dimensional lists."
            )

        num_evaluated = _to_python(
            validation["num_evaluated"]
        )

        num_correct = _to_python(
            validation["num_correct"]
        )

        if (
            isinstance(num_evaluated, bool)
            or not isinstance(num_evaluated, int)
            or num_evaluated != len(expected_fold_slides)
        ):
            raise ValueError(
                f"Fold {fold_data['fold']} has an incorrect "
                "num_evaluated value."
            )

        if (
            isinstance(num_correct, bool)
            or not isinstance(num_correct, int)
            or not 0 <= num_correct <= num_evaluated
        ):
            raise ValueError(
                "Validation num_correct is invalid."
            )

        if not (
            len(slide_ids)
            == len(true_labels)
            == len(predicted_labels)
            == len(probabilities)
            == num_evaluated
        ):
            raise ValueError(
                "Final validation output lengths are inconsistent."
            )

        if not all(
            isinstance(slide_id, str) and slide_id.strip()
            for slide_id in slide_ids
        ):
            raise TypeError(
                "Validation slide IDs must be non-empty strings."
            )

        if len(set(slide_ids)) != len(slide_ids):
            raise RuntimeError(
                "Out-of-fold predictions contain duplicate slide IDs."
            )

        duplicate_oof_ids = seen_validation_slide_ids.intersection(
            slide_ids
        )

        if duplicate_oof_ids:
            raise RuntimeError(
                "Out-of-fold predictions contain duplicate slide IDs."
            )

        seen_validation_slide_ids.update(slide_ids)

        if set(slide_ids) != expected_fold_slides:
            raise RuntimeError(
                f"Fold {fold_data['fold']} validation predictions "
                "do not match its assigned validation slides."
            )

        normalized_probabilities = []
        normalized_true_labels = []
        normalized_predicted_labels = []

        for slide_id, probability, true_label, predicted_label in zip(
            slide_ids,
            probabilities,
            true_labels,
            predicted_labels,
        ):
            expected_patient, expected_label, expected_case = (
                slide_metadata[slide_id]
            )

            probability = _finite_number(
                probability,
                "OOF probability",
            )

            if (
                probability is None
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    "OOF probabilities must be between 0 and 1."
                )

            true_label = _to_python(true_label)
            predicted_label = _to_python(predicted_label)

            if (
                isinstance(true_label, bool)
                or true_label not in (0, 1)
            ):
                raise ValueError(
                    "OOF true labels must be binary."
                )

            if int(true_label) != expected_label:
                raise ValueError(
                    f"OOF ground-truth mismatch for slide {slide_id}."
                )

            if (
                isinstance(predicted_label, bool)
                or predicted_label not in (0, 1)
            ):
                raise ValueError(
                    "OOF predicted labels must be binary."
                )

            expected_prediction = int(
                probability >= threshold
            )

            if int(predicted_label) != expected_prediction:
                raise ValueError(
                    f"OOF prediction for slide {slide_id} "
                    "does not match the configured threshold."
                )

            normalized_probabilities.append(probability)
            normalized_true_labels.append(int(true_label))
            normalized_predicted_labels.append(int(predicted_label))

        calculated_correct = sum(
            predicted == actual
            for predicted, actual in zip(
                normalized_predicted_labels,
                normalized_true_labels,
            )
        )

        if calculated_correct != num_correct:
            raise ValueError(
                "Validation num_correct does not match predictions."
            )

        case_ids = _to_python(
            validation.get("case_ids")
        )

        if any(
            slide_metadata[slide_id][2] is not None
            for slide_id in slide_ids
        ):
            if (
                not isinstance(case_ids, list)
                or len(case_ids) != num_evaluated
            ):
                raise ValueError(
                    "Validation must return case_ids for all evaluated slides."
                )

            if not all(
                isinstance(case_id, str) and case_id.strip()
                for case_id in case_ids
            ):
                raise ValueError(
                    "Validation case_ids must be non-empty strings."
                )

            for slide_id, case_id in zip(
                slide_ids,
                case_ids,
            ):
                if case_id != slide_metadata[slide_id][2]:
                    raise ValueError(
                        f"OOF case_id mismatch for slide {slide_id}."
                    )

        validation_loss = _finite_number(
            validation["mean_loss"],
            "validation mean_loss",
        )

        if validation_loss is None or validation_loss < 0:
            raise ValueError(
                "Validation mean_loss must be non-negative."
            )

        metrics = validation["metrics"]

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
                f"Validation metrics are missing: "
                f"{sorted(missing_metrics)}."
            )

        normalized_metrics = {}

        for metric_name in required_metrics:
            metric_value = _finite_number(
                metrics[metric_name],
                f"validation {metric_name}",
                allow_none=(metric_name == "auroc"),
            )

            if (
                metric_value is not None
                and not 0.0 <= metric_value <= 1.0
            ):
                raise ValueError(
                    f"Validation {metric_name} must be between 0 and 1."
                )

            normalized_metrics[metric_name] = metric_value

        metric_status = _to_python(
            validation.get("metric_status", "ok")
        )

        if not isinstance(metric_status, str):
            raise TypeError(
                "metric_status must be a string."
            )

        checkpoint_path = _to_python(
            result["checkpoint_path"]
        )

        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path.strip()
        ):
            raise ValueError(
                "checkpoint_path must be a non-empty string."
            )

        normalized_validation = {
            "mean_loss": validation_loss,
            "num_evaluated": num_evaluated,
            "num_correct": num_correct,
            "probabilities": normalized_probabilities,
            "true_labels": normalized_true_labels,
            "predicted_labels": normalized_predicted_labels,
            "slide_ids": slide_ids,
            "case_ids": case_ids,
            "metrics": normalized_metrics,
            "metric_status": metric_status,
            "diagnostics": _json_compatible(
                validation.get("diagnostics", {}),
                "validation diagnostics",
            ),
        }

        if not isinstance(
            normalized_validation["diagnostics"],
            dict,
        ):
            raise TypeError(
                "Validation diagnostics must be a mapping."
            )

        fold_result = dict(result)
        fold_result["final_train"] = {
            "loss": final_train_loss,
        }
        fold_result["final_validation"] = normalized_validation
        fold_result["checkpoint_path"] = checkpoint_path

        fold_results.append(fold_result)

    oof_predictions = []
    seen_oof_slide_ids = set()

    for result in fold_results:
        validation = result["final_validation"]

        for index, slide_id in enumerate(
            validation["slide_ids"]
        ):
            if slide_id in seen_oof_slide_ids:
                raise RuntimeError(
                    "Out-of-fold predictions contain duplicate slide IDs."
                )

            seen_oof_slide_ids.add(slide_id)

            record = {
                "slide_id": slide_id,
                "fold": result["fold"],
                "true_label": validation["true_labels"][index],
                "predicted_label": validation["predicted_labels"][index],
                "probability": validation["probabilities"][index],
            }

            if validation["case_ids"] is not None:
                record["case_id"] = validation["case_ids"][index]

            oof_predictions.append(record)

    if seen_oof_slide_ids != outer_pool:
        missing_slides = outer_pool.difference(
            seen_oof_slide_ids
        )

        extra_slides = seen_oof_slide_ids.difference(
            outer_pool
        )

        raise RuntimeError(
            "Out-of-fold predictions do not cover the exact "
            "outer training pool: "
            f"missing={len(missing_slides)}, "
            f"extra={len(extra_slides)}."
        )

    if len(oof_predictions) != len(outer_pool):
        raise RuntimeError(
            "OOF prediction count does not match the outer training pool."
        )

    fold_summaries = []

    for result in fold_results:
        validation = result["final_validation"]

        fold_summaries.append(
            {
                "fold": result["fold"],
                "epochs_completed": result["epochs_completed"],
                "train_slides": result["train_slides"],
                "validation_slides": result["validation_slides"],
                "final_train_loss": result["final_train"]["loss"],
                "validation_loss": validation["mean_loss"],
                "num_evaluated": validation["num_evaluated"],
                "num_correct": validation["num_correct"],
                "metrics": validation["metrics"],
                "metric_status": validation["metric_status"],
                "checkpoint_path": result["checkpoint_path"],
            }
        )

    metric_names = (
        "accuracy",
        "auroc",
        "recall",
        "f1",
    )

    summary = {
        "training_loss": _aggregate(
            [
                item["final_train_loss"]
                for item in fold_summaries
            ]
        ),
        "validation_loss": _aggregate(
            [
                item["validation_loss"]
                for item in fold_summaries
            ]
        ),
        "metrics": {
            name: _aggregate(
                [
                    item["metrics"][name]
                    for item in fold_summaries
                ]
            )
            for name in metric_names
        },
        "num_folds": 5,
        "num_oof_slides": len(oof_predictions),
        "fold_selection": "not_performed",
    }

    payload = {
        "protocol": {
            "seed": seed,
            "num_folds": 5,
            "epochs_per_fold": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(resolved_device),
            "eps": float(eps),
            "threshold": float(threshold),
            "config": config_payload,
            "training_views": [
                "Original",
                "Aug1",
                "Aug2",
            ],
            "validation_view": "Original",
            "epoch_policy": "fixed_epochs",
            "early_stopping": False,
            "best_checkpoint_selection": False,
            "test_set_used": False,
        },
        "summary": summary,
        "fold_summaries": fold_summaries,
        "oof_predictions": oof_predictions,
        "fold_results": fold_results,
    }

    if results_path is not None:
        destination = _write_json_atomic(
            results_path,
            payload,
        )

        payload["results_path"] = str(destination)

    return payload
