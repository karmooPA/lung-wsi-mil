"""Fit one fold, select by validation AUROC/loss, and resume exactly by epoch."""

from pathlib import Path

import torch
from torch.optim import Adam

from lung_wsi_mil.data.mil_dataloader import create_dataset_subset, create_mil_dataloader
from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
from lung_wsi_mil.experiments.artifacts import (
    atomic_checkpoint,
    atomic_csv,
    atomic_json,
    hash_file,
    load_checkpoint,
    read_json,
    seal_stage,
    verify_stage,
)
from lung_wsi_mil.experiments.reproducibility import (
    capture_rng,
    derived_seed,
    restore_rng,
    seed_everything,
    seed_worker,
)
from lung_wsi_mil.mil.amil import AMIL
from lung_wsi_mil.mil.evaluation import evaluate_model
from lung_wsi_mil.mil.training import train_one_epoch


def is_improved(candidate, best):
    """Keep the actual maximum AUROC, using exact loss ordering only for ties."""
    if best is None:
        return True
    return candidate["auroc"] > best["auroc"] or (
        candidate["auroc"] == best["auroc"] and candidate["loss"] < best["loss"]
    )


def meaningful_progress(candidate, reference, min_delta):
    """min_delta controls patience resets, without changing checkpoint ranking."""
    if reference is None:
        return True
    return candidate["auroc"] > reference["auroc"] + min_delta or (
        candidate["auroc"] == reference["auroc"]
        and candidate["loss"] < reference["loss"] - min_delta
    )


def prediction_metadata(context, seed, fold, checkpoint_sha256):
    return {
        "run_id": context["fingerprint"],
        "seed": seed,
        "fold": fold,
        "model_id": "AMIL",
        "checkpoint_sha256": checkpoint_sha256,
        "config_hash": context["config_hash"],
        "cohort_hash": context["inventory_sha256"],
        "split_hash": context["split_hash"],
        "feature_sha256": context["features"]["sha256"],
    }


def _best_payload(config, context, seed, fold, best_state, best_epoch, best_metrics):
    return {
        "schema_version": 1,
        "run_fingerprint": context["fingerprint"],
        "seed": seed,
        "fold": fold,
        "config": config.to_dict(),
        "epoch": best_epoch,
        "model_state": best_state,
        "validation": best_metrics,
    }


def fit_fold(features, train_frame, val_frame, directory, config, context, *, seed, fold):
    directory = Path(directory)
    if (directory / "complete.json").exists():
        verify_stage(directory, run_fingerprint=context["fingerprint"])
        return read_json(directory / "summary.json")
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.tmp.*"):
        path.unlink()
    seed_everything(derived_seed(seed, fold))
    dataset = MILHDF5Dataset(features)
    generators = {
        name: torch.Generator().manual_seed(derived_seed(seed, fold, stream))
        for stream, name in enumerate(("train", "validation"), 1)
    }
    train_loader = create_mil_dataloader(
        create_dataset_subset(dataset, train_frame),
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generators["train"],
        worker_init_fn=seed_worker,
    )
    val_loader = create_mil_dataloader(
        create_dataset_subset(dataset, val_frame),
        config.batch_size,
        num_workers=config.num_workers,
        view_indices=(0,),
        generator=generators["validation"],
        worker_init_fn=seed_worker,
    )
    device = torch.device(config.device)
    model = AMIL(attention_dim=config.attention_dim, threshold=config.threshold).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_metrics, best_state = [], None, None
    best_epoch, bad_epochs, start_epoch = 0, 0, 1
    patience_reference = None
    last_path = directory / "last.pt"
    if last_path.exists():
        state = load_checkpoint(
            last_path, run_fingerprint=context["fingerprint"], seed=seed, fold=fold
        )
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        history, best_metrics, best_state = (
            state["history"],
            state["best_metrics"],
            state["best_model_state"],
        )
        best_epoch, bad_epochs, start_epoch = (
            state["best_epoch"],
            state["bad_epochs"],
            state["epoch"] + 1,
        )
        patience_reference = state["patience_reference"]
        if len(history) != state["epoch"] or not 1 <= best_epoch <= state["epoch"] <= config.epochs:
            raise ValueError("Checkpoint epoch/history metadata is inconsistent.")
        restore_rng(state["rng"], generators)
    for epoch in range(start_epoch, config.epochs + 1):
        if bad_epochs >= config.patience:
            break
        train = train_one_epoch(model, train_loader, optimizer)
        validation = evaluate_model(model, val_loader, threshold=config.threshold)
        candidate = {"auroc": validation["metrics"]["auroc"], "loss": validation["loss"]}
        if candidate["auroc"] is None:
            raise ValueError("Validation AUROC is undefined; fix the patient partition.")
        if meaningful_progress(candidate, patience_reference, config.min_delta):
            patience_reference, bad_epochs = candidate, 0
        else:
            bad_epochs += 1
        if is_improved(candidate, best_metrics):
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            best_metrics, best_epoch = candidate, epoch
        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_bags": train["num_bags"],
                "train_slides": train["num_slides"],
                "validation_loss": validation["loss"],
                "validation_auroc": candidate["auroc"],
                "bad_epochs": bad_epochs,
            }
        )
        state = {
            "schema_version": 1,
            "run_fingerprint": context["fingerprint"],
            "seed": seed,
            "fold": fold,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_model_state": best_state,
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "patience_reference": patience_reference,
            "history": history,
            "rng": capture_rng(generators),
        }
        # last.pt is the atomic source of truth, including the best state. A
        # crash between this write and best.pt/history writes is recoverable.
        atomic_checkpoint(last_path, state)
        atomic_checkpoint(
            directory / "best.pt",
            _best_payload(config, context, seed, fold, best_state, best_epoch, best_metrics),
        )
        atomic_json(directory / "history.json", history)
        print(
            f"seed={seed} fold={fold} epoch={epoch} train_loss={train['loss']:.6f} "
            f"val_auc={candidate['auroc']:.6f} val_loss={candidate['loss']:.6f}",
            flush=True,
        )
    if best_state is None:
        raise ValueError("Fold did not produce a selected checkpoint.")
    atomic_checkpoint(
        directory / "best.pt",
        _best_payload(config, context, seed, fold, best_state, best_epoch, best_metrics),
    )
    atomic_json(directory / "history.json", history)
    model.load_state_dict(best_state, strict=True)
    selected = evaluate_model(model, val_loader, threshold=config.threshold)
    if (
        selected["metrics"]["auroc"] != best_metrics["auroc"]
        or abs(selected["loss"] - best_metrics["loss"]) > 1e-8
    ):
        raise ValueError("Reloaded best checkpoint does not reproduce its validation scores.")
    checkpoint_hash = hash_file(directory / "best.pt")
    provenance = prediction_metadata(context, seed, fold, checkpoint_hash)
    atomic_csv(
        directory / "validation_predictions.csv",
        [{**provenance, **row} for row in selected["predictions"]],
    )
    atomic_json(directory / "validation_metrics.json", selected["metrics"])
    summary = {
        "seed": seed,
        "fold": fold,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": len(history) < config.epochs,
        "validation": {**selected["metrics"], "loss": selected["loss"]},
        "checkpoint_sha256": checkpoint_hash,
        "train_slides": len(train_frame),
        "validation_slides": len(val_frame),
        "test_used_for_selection": False,
    }
    atomic_json(directory / "summary.json", summary)
    seal_stage(
        directory, run_fingerprint=context["fingerprint"], details={"seed": seed, "fold": fold}
    )
    return summary
