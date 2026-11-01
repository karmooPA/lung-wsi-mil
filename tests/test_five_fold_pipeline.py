
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

import src.mil.fold_runner as fold_runner
import src.mil.five_fold_runner as five_fold_runner


class FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset


class FakeCheckpointManager:
    instances = []

    def __init__(self, root, seed, fold):
        self.root = Path(root)
        self.seed = seed
        self.fold = fold
        self.saved = []
        self.instances.append(self)

    def save(self, name, model, optimizer, **kwargs):
        self.saved.append(
            {
                "name": name,
                "epoch": kwargs["epoch"],
                "model": model,
                "optimizer": optimizer,
                "kwargs": kwargs,
            }
        )

    def path_for(self, name):
        return (
            self.root
            / f"seed_{self.seed}"
            / f"fold_{self.fold}"
            / f"{name}.pt"
        )


def _make_simple_fold():
    train = pd.DataFrame(
        [
            {
                "slide_id": f"TR_{i}",
                "patient_id": f"TRP_{i}",
                "label": i % 2,
            }
            for i in range(4)
        ]
    )

    val = pd.DataFrame(
        [
            {
                "slide_id": f"VA_{i}",
                "patient_id": f"VAP_{i}",
                "label": i % 2,
            }
            for i in range(2)
        ]
    )

    return {
        "fold": 1,
        "train": train,
        "val": val,
    }


def _make_dataset(*frames):
    slide_ids = sorted(
        {
            slide_id
            for frame in frames
            for slide_id in frame["slide_id"].tolist()
        }
    )

    return SimpleNamespace(slide_ids=slide_ids)


def _loader_factory(
    *,
    dataset,
    fold_train_df,
    fold_val_df,
    batch_size,
    num_workers,
    seed,
):
    return (
        FakeLoader(fold_train_df.copy()),
        FakeLoader(fold_val_df.copy()),
    )


def _validation_output(frame):
    labels = [
        int(value)
        for value in frame["label"].tolist()
    ]

    slide_ids = frame["slide_id"].tolist()

    probabilities = [
        0.9 if label == 1 else 0.1
        for label in labels
    ]

    return {
        "mean_loss": 0.4,
        "num_evaluated": len(slide_ids),
        "num_correct": len(slide_ids),
        "probabilities": probabilities,
        "true_labels": labels,
        "predicted_labels": labels.copy(),
        "slide_ids": slide_ids,
        "metrics": {
            "accuracy": 1.0,
            "auroc": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "metric_status": "ok",
        "diagnostics": {},
    }


def _validation_fn(model, val_loader, device, threshold):
    return _validation_output(val_loader.dataset)


def _fake_train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    eps,
):
    num_slides = len(dataloader.dataset)

    return {
        "loss": 0.5,
        "num_batches": 1,
        "num_bags": 3 * num_slides,
        "num_slides": num_slides,
    }


@pytest.fixture
def fold_environment(monkeypatch):
    FakeCheckpointManager.instances.clear()

    monkeypatch.setattr(
        fold_runner,
        "CheckpointManager",
        FakeCheckpointManager,
    )

    monkeypatch.setattr(
        fold_runner,
        "train_one_epoch",
        _fake_train_one_epoch,
    )


def _call_fold(
    tmp_path,
    fold_data=None,
    validation_fn=None,
    dataloader_factory=None,
    optimizer_factory=None,
    epochs=2,
):
    fold_data = fold_data or _make_simple_fold()

    dataset = _make_dataset(
        fold_data["train"],
        fold_data["val"],
    )

    return fold_runner.run_fold(
        fold_data,
        seed=2,
        dataset=dataset,
        batch_size=2,
        model_factory=lambda: nn.Linear(1, 1),
        optimizer_factory=(
            optimizer_factory
            or (
                lambda params: torch.optim.SGD(
                    params,
                    lr=0.01,
                )
            )
        ),
        validation_fn=validation_fn or _validation_fn,
        checkpoint_root=tmp_path,
        epochs=epochs,
        device="cpu",
        dataloader_factory=dataloader_factory or _loader_factory,
    )


def test_fold_runner_completes_all_epochs(
    tmp_path,
    fold_environment,
):
    result = _call_fold(
        tmp_path,
        epochs=3,
    )

    assert result["seed"] == 2
    assert result["fold"] == 1
    assert result["epochs_completed"] == 3
    assert len(result["history"]) == 3
    assert result["final_train"]["loss"] == 0.5
    assert result["final_validation"]["mean_loss"] == 0.4


def test_fold_runner_saves_checkpoint_after_each_epoch(
    tmp_path,
    fold_environment,
):
    _call_fold(
        tmp_path,
        epochs=3,
    )

    manager = FakeCheckpointManager.instances[-1]

    assert len(manager.saved) == 3
    assert [
        item["epoch"]
        for item in manager.saved
    ] == [1, 2, 3]

    assert all(
        item["name"] == "last"
        for item in manager.saved
    )


def test_fold_runner_returns_correct_checkpoint_path(
    tmp_path,
    fold_environment,
):
    result = _call_fold(tmp_path)

    assert result["checkpoint_path"].endswith(
        "seed_2/fold_1/last.pt"
    )


def test_fold_runner_calls_validation_after_each_epoch(
    tmp_path,
    fold_environment,
):
    calls = []

    def validation_fn(model, val_loader, device, threshold):
        calls.append(len(val_loader.dataset))
        return _validation_output(val_loader.dataset)

    _call_fold(
        tmp_path,
        validation_fn=validation_fn,
        epochs=3,
    )

    assert calls == [2, 2, 2]


def test_fold_runner_rejects_patient_leakage(
    tmp_path,
    fold_environment,
):
    fold = _make_simple_fold()

    fold["val"].loc[0, "patient_id"] = (
        fold["train"].iloc[0]["patient_id"]
    )

    with pytest.raises(ValueError, match="Patient leakage"):
        _call_fold(tmp_path, fold_data=fold)


def test_fold_runner_rejects_slide_leakage(
    tmp_path,
    fold_environment,
):
    fold = _make_simple_fold()

    fold["val"].loc[0, "slide_id"] = (
        fold["train"].iloc[0]["slide_id"]
    )

    with pytest.raises(ValueError, match="Slide leakage"):
        _call_fold(tmp_path, fold_data=fold)


def test_fold_runner_rejects_invalid_fold_number(
    tmp_path,
    fold_environment,
):
    fold = _make_simple_fold()
    fold["fold"] = 6

    with pytest.raises(ValueError, match="Fold number"):
        _call_fold(tmp_path, fold_data=fold)


def test_fold_runner_rejects_invalid_labels(
    tmp_path,
    fold_environment,
):
    fold = _make_simple_fold()
    fold["train"].loc[0, "label"] = 2

    with pytest.raises(ValueError, match="non-binary"):
        _call_fold(tmp_path, fold_data=fold)


def test_fold_runner_rejects_incorrect_training_bag_count(
    tmp_path,
    fold_environment,
    monkeypatch,
):
    def invalid_train(*args, **kwargs):
        return {
            "loss": 0.5,
            "num_batches": 1,
            "num_bags": 11,
            "num_slides": 4,
        }

    monkeypatch.setattr(
        fold_runner,
        "train_one_epoch",
        invalid_train,
    )

    with pytest.raises(ValueError, match="expected 12"):
        _call_fold(tmp_path)


def test_fold_runner_rejects_missing_validation_slide(
    tmp_path,
    fold_environment,
):
    def invalid_validation(model, val_loader, device, threshold):
        frame = val_loader.dataset.iloc[:1]
        return _validation_output(frame)

    with pytest.raises(ValueError, match="coverage mismatch"):
        _call_fold(
            tmp_path,
            validation_fn=invalid_validation,
        )


def test_fold_runner_rejects_duplicate_validation_slide(
    tmp_path,
    fold_environment,
):
    def invalid_validation(model, val_loader, device, threshold):
        result = _validation_output(val_loader.dataset)
        result["slide_ids"][1] = result["slide_ids"][0]
        return result

    with pytest.raises(ValueError, match="Duplicate validation"):
        _call_fold(
            tmp_path,
            validation_fn=invalid_validation,
        )


def test_fold_runner_rejects_invalid_probability(
    tmp_path,
    fold_environment,
):
    def invalid_validation(model, val_loader, device, threshold):
        result = _validation_output(val_loader.dataset)
        result["probabilities"][0] = 1.5
        return result

    with pytest.raises(ValueError, match="between 0 and 1"):
        _call_fold(
            tmp_path,
            validation_fn=invalid_validation,
        )


def test_fold_runner_rejects_three_dataloaders(
    tmp_path,
    fold_environment,
):
    def invalid_factory(**kwargs):
        return (
            FakeLoader(kwargs["fold_train_df"]),
            FakeLoader(kwargs["fold_val_df"]),
            FakeLoader(kwargs["fold_val_df"]),
        )

    with pytest.raises(ValueError, match="exactly"):
        _call_fold(
            tmp_path,
            dataloader_factory=invalid_factory,
        )


def test_fold_runner_rejects_optimizer_from_another_model(
    tmp_path,
    fold_environment,
):
    def invalid_optimizer_factory(params):
        other_model = nn.Linear(1, 1)

        return torch.optim.SGD(
            other_model.parameters(),
            lr=0.01,
        )

    with pytest.raises(ValueError, match="parameters of this model"):
        _call_fold(
            tmp_path,
            optimizer_factory=invalid_optimizer_factory,
        )


def _make_five_folds():
    rows = []

    for patient_index in range(10):
        patient_id = f"P{patient_index:02d}"
        label = patient_index % 2

        for slide_index in range(2):
            rows.append(
                {
                    "slide_id": f"S{patient_index:02d}_{slide_index}",
                    "patient_id": patient_id,
                    "label": label,
                }
            )

    metadata = pd.DataFrame(rows)
    folds = []

    for fold_number in range(1, 6):
        val_patients = {
            f"P{index:02d}"
            for index in range(
                (fold_number - 1) * 2,
                fold_number * 2,
            )
        }

        val = metadata[
            metadata["patient_id"].isin(val_patients)
        ].copy().reset_index(drop=True)

        train = metadata[
            ~metadata["patient_id"].isin(val_patients)
        ].copy().reset_index(drop=True)

        folds.append(
            {
                "fold": fold_number,
                "train": train,
                "val": val,
            }
        )

    return folds, metadata


def _fake_run_fold_factory(
    models,
    optimizers,
    calls,
    duplicate_oof=False,
    folds=None,
):
    def fake_run_fold(fold_data, **kwargs):
        model = kwargs["model_factory"]()
        models.append(model)

        optimizer = kwargs["optimizer_factory"](
            model.parameters()
        )
        optimizers.append(optimizer)

        frame = fold_data["val"]

        if duplicate_oof and fold_data["fold"] == 2:
            frame = folds[0]["val"]

        validation = _validation_output(frame)

        calls.append(
            {
                "seed": kwargs["seed"],
                "fold": fold_data["fold"],
                "dataset": kwargs["dataset"],
                "batch_size": kwargs["batch_size"],
                "epochs": kwargs["epochs"],
                "dataloader_factory": kwargs["dataloader_factory"],
            }
        )

        return {
            "seed": kwargs["seed"],
            "fold": fold_data["fold"],
            "epochs_completed": kwargs["epochs"],
            "train_slides": len(fold_data["train"]),
            "validation_slides": len(fold_data["val"]),
            "final_train": {
                "loss": 0.5,
            },
            "final_validation": validation,
            "history": [],
            "checkpoint_path": (
                f"seed_{kwargs['seed']}/fold_{fold_data['fold']}/last.pt"
            ),
        }

    return fake_run_fold


def _call_five_folds(
    tmp_path,
    folds,
    dataset,
    results_path=None,
):
    return five_fold_runner.run_five_folds(
        folds,
        seed=2,
        dataset=dataset,
        batch_size=2,
        model_factory=lambda: nn.Linear(1, 1),
        optimizer_factory=lambda params: torch.optim.SGD(
            params,
            lr=0.01,
        ),
        validation_fn=_validation_fn,
        checkpoint_root=tmp_path / "checkpoints",
        epochs=2,
        device="cpu",
        results_path=results_path,
    )


def test_five_fold_runner_executes_exactly_five_folds(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert len(result["fold_results"]) == 5
    assert len(result["fold_summaries"]) == 5

    assert [
        item["fold"]
        for item in result["fold_summaries"]
    ] == [1, 2, 3, 4, 5]

    assert len(calls) == 5


def test_five_fold_runner_creates_independent_models_and_optimizers(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert len(models) == 5
    assert len({id(model) for model in models}) == 5

    assert len(optimizers) == 5
    assert len({id(optimizer) for optimizer in optimizers}) == 5


def test_five_fold_runner_preserves_single_seed(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert result["protocol"]["seed"] == 2
    assert all(
        call["seed"] == 2
        for call in calls
    )


def test_five_fold_runner_collects_all_oof_predictions(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert len(result["oof_predictions"]) == len(metadata)

    predicted_slide_ids = {
        item["slide_id"]
        for item in result["oof_predictions"]
    }

    assert predicted_slide_ids == set(metadata["slide_id"])


def test_five_fold_runner_aggregates_fold_metrics(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert result["summary"]["num_folds"] == 5
    assert result["summary"]["num_oof_slides"] == 20

    accuracy = result["summary"]["metrics"]["accuracy"]

    assert accuracy["mean"] == 1.0
    assert accuracy["std"] == 0.0
    assert accuracy["n_defined"] == 5


def test_five_fold_runner_does_not_use_final_test(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert result["protocol"]["test_set_used"] is False

    for call in calls:
        assert call["dataset"] is dataset


def test_five_fold_runner_uses_fixed_epochs(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
    )

    assert result["protocol"]["epoch_policy"] == "fixed_epochs"
    assert result["protocol"]["early_stopping"] is False

    assert (
        result["protocol"]["best_checkpoint_selection"]
        is False
    )


def test_five_fold_runner_writes_results_json(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
        ),
    )

    results_path = tmp_path / "results" / "seed_2.json"

    result = _call_five_folds(
        tmp_path,
        folds,
        dataset,
        results_path=results_path,
    )

    assert results_path.exists()
    assert result["results_path"] == str(results_path)

    saved = json.loads(
        results_path.read_text(encoding="utf-8")
    )

    assert saved["protocol"]["seed"] == 2
    assert len(saved["fold_summaries"]) == 5
    assert len(saved["oof_predictions"]) == 20


def test_five_fold_runner_rejects_missing_fold(
    tmp_path,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    with pytest.raises(ValueError, match="Exactly five folds"):
        _call_five_folds(
            tmp_path,
            folds[:4],
            dataset,
        )


def test_five_fold_runner_rejects_duplicate_fold_number(
    tmp_path,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    folds[1]["fold"] = 1

    with pytest.raises(ValueError, match="appears more than once"):
        _call_five_folds(
            tmp_path,
            folds,
            dataset,
        )


def test_five_fold_runner_rejects_patient_leakage(
    tmp_path,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    leaked_row = folds[0]["val"].iloc[[0]].copy()
    leaked_row.loc[:, "slide_id"] = "LEAKED_SLIDE"

    folds[0]["train"] = pd.concat(
        [
            folds[0]["train"],
            leaked_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Patient leakage"):
        _call_five_folds(
            tmp_path,
            folds,
            dataset,
        )


def test_five_fold_runner_rejects_patient_across_validation_folds(
    tmp_path,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    replacement_patients = {"P00", "P03"}

    folds[1]["val"] = metadata[
        metadata["patient_id"].isin(replacement_patients)
    ].copy().reset_index(drop=True)

    folds[1]["train"] = metadata[
        ~metadata["patient_id"].isin(replacement_patients)
    ].copy().reset_index(drop=True)

    with pytest.raises(ValueError, match="multiple validation folds"):
        _call_five_folds(
            tmp_path,
            folds,
            dataset,
        )


def test_five_fold_runner_rejects_incomplete_training_complement(
    tmp_path,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    folds[0]["train"] = (
        folds[0]["train"]
        .iloc[1:]
        .copy()
        .reset_index(drop=True)
    )

    with pytest.raises(ValueError, match="same outer training pool"):
        _call_five_folds(
            tmp_path,
            folds,
            dataset,
        )


def test_five_fold_runner_rejects_duplicate_oof_predictions(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        _fake_run_fold_factory(
            models,
            optimizers,
            calls,
            duplicate_oof=True,
            folds=folds,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate slide IDs",
    ):
        _call_five_folds(
            tmp_path,
            folds,
            dataset,
        )


def test_five_fold_runner_rejects_reused_model_factory(
    tmp_path,
    monkeypatch,
):
    folds, metadata = _make_five_folds()
    dataset = _make_dataset(metadata)

    models = []
    optimizers = []
    calls = []

    fake_run_fold = _fake_run_fold_factory(
        models,
        optimizers,
        calls,
    )

    singleton = nn.Linear(1, 1)

    def reused_model_factory():
        return singleton

    monkeypatch.setattr(
        five_fold_runner,
        "run_fold",
        fake_run_fold,
    )

    with pytest.raises(ValueError, match="fresh object"):
        five_fold_runner.run_five_folds(
            folds,
            seed=2,
            dataset=dataset,
            batch_size=2,
            model_factory=reused_model_factory,
            optimizer_factory=lambda params: torch.optim.SGD(
                params,
                lr=0.01,
            ),
            validation_fn=_validation_fn,
            checkpoint_root=tmp_path / "checkpoints",
            epochs=2,
        )
