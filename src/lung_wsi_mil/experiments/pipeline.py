"""Frozen 5×5 training and a separate final held-out test stage."""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lung_wsi_mil.experiments.artifacts import (
    atomic_csv,
    atomic_json,
    fingerprint,
    hash_file,
    read_json,
    run_lock,
    seal_stage,
    verify_stage,
)
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.fold import fit_fold, prediction_metadata
from lung_wsi_mil.experiments.inputs import (
    REFERENCE_INVENTORY_SHA256,
    load_metadata,
    validate_features,
    validate_partitions,
)
from lung_wsi_mil.experiments.reproducibility import configure_runtime
from lung_wsi_mil.paths import PACKAGE_ROOT, PROJECT_ROOT


def code_identity():
    roots = ("data", "mil", "evaluation", "experiments", "visualization")
    paths = [PACKAGE_ROOT / "paths.py"]
    for name in roots:
        paths.extend((PACKAGE_ROOT / name).rglob("*.py"))
    files = {
        path.relative_to(PACKAGE_ROOT).as_posix(): hash_file(path) for path in sorted(set(paths))
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"file_sha256": files, "git_commit": commit, "git_dirty": dirty}


def _validate_output(output, features):
    if output.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(output):
        raise ValueError("Experiment output must be outside the source repository.")
    if features.is_relative_to(output):
        raise ValueError("Experiment output must not contain its feature input.")


def _assert_feature_stable(path, identity, *, hash_contents=False):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (identity["size_bytes"], identity["mtime_ns"]):
        raise ValueError("Feature input changed during the experiment.")
    if hash_contents and hash_file(path) != identity["sha256"]:
        raise ValueError("Feature contents changed during the experiment.")


def _status(output, status, **extra):
    atomic_json(
        Path(output) / "status.json",
        {"status": status, "updated_utc": datetime.now(UTC).isoformat(), **extra},
    )


def _check_manifest(manifest):
    identity = manifest["identity"]
    if fingerprint(identity) != manifest.get("fingerprint"):
        raise ValueError("Run identity checksum mismatch.")
    expected_hashes = {"inventory.csv": identity["inventory_sha256"], **identity["split_hashes"]}
    if (
        manifest["features"]["sha256"] != identity["feature_sha256"]
        or manifest["inventory_sha256"] != identity["inventory_sha256"]
        or manifest["split_hash"] != fingerprint(identity["split_hashes"])
        or manifest["input_hashes"] != expected_hashes
    ):
        raise ValueError("Run input references differ from the frozen identity.")
    if (
        fingerprint(manifest["config"]) != manifest["config_hash"]
        or manifest["config"] != identity["config"]
    ):
        raise ValueError("Frozen configuration checksum mismatch.")


def _snapshot_inputs(output, inventory, splits, hashes):
    snapshots = output / "inputs"
    for relative, expected in hashes.items():
        source = (
            inventory if relative == "inventory.csv" else splits / relative.removeprefix("splits/")
        )
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hash_file(target) != expected:
                raise ValueError(f"Stored input snapshot differs: {relative}")
        else:
            shutil.copyfile(source, target)
        if hash_file(target) != expected:
            raise ValueError(f"Input changed while snapshotting: {relative}")


def _verify_snapshots(output, manifest):
    for relative, expected in manifest["input_hashes"].items():
        path = (output / "inputs" / relative).resolve()
        if (
            not path.is_relative_to((output / "inputs").resolve())
            or not path.is_file()
            or hash_file(path) != expected
        ):
            raise ValueError(f"Missing or modified input snapshot: {relative}")


def select_final_fold(summaries):
    """Selection uses validation only: AUROC descending, loss ascending, fold ascending."""
    if not summaries or any(item["validation"].get("auroc") is None for item in summaries):
        raise ValueError("All candidate folds require a defined validation AUROC.")
    return min(
        summaries,
        key=lambda item: (-item["validation"]["auroc"], item["validation"]["loss"], item["fold"]),
    )


def train_experiment(*, features, inventory, splits, output, config=None, resume=False):
    """Preflight every fold, freeze inputs/protocol, then fit without test inference."""
    features, inventory, splits, output = [
        Path(path).expanduser().resolve() for path in (features, inventory, splits, output)
    ]
    _validate_output(output, features)
    if output.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {output}; use --resume only for this same experiment."
        )
    prior = read_json(output / "run.json") if resume else None
    if prior is not None:
        _check_manifest(prior)
    config = config or (
        ExperimentConfig.from_dict(prior["config"]) if prior else ExperimentConfig()
    )
    environment = configure_runtime(config)
    metadata = load_metadata(inventory)
    inventory_hash = hash_file(inventory)
    if inventory_hash != metadata.attrs["source_sha256"]:
        raise ValueError("Inventory changed during preflight.")
    partitions, split_hashes = validate_partitions(metadata, splits, config)
    feature_info = validate_features(
        features,
        metadata,
        inventory_sha256=inventory_hash,
        allow_unverified=config.allow_unverified_features,
    )
    code = code_identity()
    identity = {
        "config": config.to_dict(),
        "inventory_sha256": inventory_hash,
        "split_hashes": split_hashes,
        "feature_sha256": feature_info["sha256"],
        "code_sha256": code["file_sha256"],
        "environment": environment,
        "protocol": {
            "training_views": ["Original", "Aug1", "Aug2"],
            "validation_and_test_views": ["Original"],
            "checkpoint_selection": "validation_auroc_then_loss",
            "final_model_selection": "best_validation_fold_then_lowest_fold_index",
            "test_threshold": config.threshold,
            "optimizer": "Adam",
        },
    }
    run_fingerprint = fingerprint(identity)
    if prior is not None and prior.get("fingerprint") != run_fingerprint:
        raise ValueError("Resume refused: configuration, inputs, code, runtime or device differs.")
    manifest = prior or {
        "schema_version": 1,
        "fingerprint": run_fingerprint,
        "identity": identity,
        "config": config.to_dict(),
        "config_hash": fingerprint(config.to_dict()),
        "inventory_sha256": inventory_hash,
        "split_hash": fingerprint(split_hashes),
        "features": feature_info,
        "code": code,
        "created_utc": datetime.now(UTC).isoformat(),
        "input_hashes": {"inventory.csv": inventory_hash, **split_hashes},
        "source_paths": {
            "features": str(features),
            "inventory": str(inventory),
            "splits": str(splits),
        },
        "verified_reference_cohort": (
            inventory_hash == REFERENCE_INVENTORY_SHA256
            and feature_info["official_provenance_verified"]
        ),
        "hyperparameter_origin": "Explicit engineering defaults, not claimed exact paper hyperparameters.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        if not prior:
            if (output / "run.json").exists():
                raise FileExistsError(
                    "An experiment appeared during preflight; its manifest will not be overwritten."
                )
            atomic_json(output / "run.json", manifest)
        _snapshot_inputs(output, inventory, splits, manifest["input_hashes"])
        _verify_snapshots(output, manifest)
        training = output / "training"
        if (training / "complete.json").exists():
            verify_stage(training, run_fingerprint=run_fingerprint)
            return read_json(training / "summary.json")
        _status(output, "training", completed_fits=0, expected_fits=len(config.seeds) * 5)
        try:
            seeds, completed = [], 0
            for seed in config.seeds:
                summaries, predictions = [], []
                for fold in range(1, 6):
                    _assert_feature_stable(features, feature_info)
                    frames = partitions[seed]
                    directory = training / f"seed_{seed}" / f"fold_{fold}"
                    summaries.append(
                        fit_fold(
                            features,
                            frames[f"fold_{fold}_train"],
                            frames[f"fold_{fold}_val"],
                            directory,
                            config,
                            manifest,
                            seed=seed,
                            fold=fold,
                        )
                    )
                    predictions.extend(
                        pd.read_csv(directory / "validation_predictions.csv").to_dict("records")
                    )
                    completed += 1
                    _status(
                        output,
                        "training",
                        completed_fits=completed,
                        expected_fits=len(config.seeds) * 5,
                    )
                if len({row["slide_id"] for row in predictions}) != len(
                    partitions[seed]["train_cv"]
                ):
                    raise ValueError("Out-of-fold prediction coverage is incomplete.")
                atomic_csv(
                    training / f"seed_{seed}" / "validation_oof_predictions.csv",
                    sorted(predictions, key=lambda row: row["slide_id"]),
                )
                selected = select_final_fold(summaries)
                selection = {
                    "seed": seed,
                    "selected_fold": selected["fold"],
                    "selected_epoch": selected["best_epoch"],
                    "checkpoint": f"seed_{seed}/fold_{selected['fold']}/best.pt",
                    "checkpoint_sha256": selected["checkpoint_sha256"],
                    "selected_validation": selected["validation"],
                    "folds": summaries,
                    "selection_uses_test": False,
                }
                atomic_json(training / f"seed_{seed}" / "selection.json", selection)
                seeds.append(selection)
            _assert_feature_stable(features, feature_info, hash_contents=True)
            if code_identity()["file_sha256"] != code["file_sha256"]:
                raise ValueError("Experiment code changed during training.")
            summary = {
                "status": "training_complete",
                "run_fingerprint": run_fingerprint,
                "completed_fits": completed,
                "expected_fits": len(config.seeds) * 5,
                "seeds": seeds,
                "test_forward_passes": 0,
                "oof_interpretation": "Selected-epoch validation predictions; descriptive model-selection data, not the final test estimate.",
            }
            atomic_json(training / "summary.json", summary)
            seal_stage(
                training, run_fingerprint=run_fingerprint, details={"completed_fits": completed}
            )
            _status(output, "training_complete", completed_fits=completed, test_started=False)
            return summary
        except (Exception, KeyboardInterrupt) as error:
            _status(output, "training_interrupted", error=f"{type(error).__name__}: {error}")
            raise


def load_frozen_run(output, features=None):
    """Verify frozen training state before any final-test model is constructed."""
    output = Path(output).expanduser().resolve()
    manifest = read_json(output / "run.json")
    _check_manifest(manifest)
    config = ExperimentConfig.from_dict(manifest["config"])
    if fingerprint(config.to_dict()) != manifest["config_hash"]:
        raise ValueError("Frozen configuration checksum mismatch.")
    if manifest["identity"]["config"] != json.loads(json.dumps(config.to_dict())):
        raise ValueError("Frozen configuration differs from its identity.")
    if code_identity()["file_sha256"] != manifest["identity"]["code_sha256"]:
        raise ValueError("Current experiment code differs from the frozen training code.")
    environment = configure_runtime(config)
    if environment != manifest["identity"]["environment"]:
        raise ValueError("Final-test runtime/device differs from the frozen training runtime.")
    _verify_snapshots(output, manifest)
    metadata = load_metadata(output / "inputs/inventory.csv")
    partitions, hashes = validate_partitions(metadata, output / "inputs/splits", config)
    if hashes != manifest["identity"]["split_hashes"]:
        raise ValueError("Frozen split identity mismatch.")
    features = Path(features or manifest["source_paths"]["features"]).expanduser().resolve()
    info = validate_features(
        features,
        metadata,
        inventory_sha256=manifest["inventory_sha256"],
        allow_unverified=config.allow_unverified_features,
    )
    if info["sha256"] != manifest["features"]["sha256"]:
        raise ValueError("Final-test features differ from training features.")
    verify_stage(output / "training", run_fingerprint=manifest["fingerprint"])
    training = read_json(output / "training/summary.json")
    if training["completed_fits"] != len(config.seeds) * 5 or len(training["seeds"]) != len(
        config.seeds
    ):
        raise ValueError("All requested folds/seeds must finish before final testing.")
    return output, features, config, manifest, training, partitions, info


def _heatmap_slides(frame, limit):
    if limit == 0 or limit >= len(frame):
        return sorted(frame.slide_id)
    chosen = []
    for label in (0, 1):
        chosen.extend(frame[frame.label == label].slide_id.tolist()[: limit // 2])
    chosen.extend(slide for slide in frame.slide_id if slide not in chosen)
    return chosen[:limit]


def _rendering_spec(patch_root, partitions, config, features):
    if patch_root is None:
        return {
            "requested": False,
            "reason": "No preprocessing root supplied; all raw attention is still exported.",
        }
    sources = {}
    with h5py.File(features, "r") as file:
        for seed in config.seeds:
            for slide in _heatmap_slides(partitions[seed]["test"], config.heatmap_limit):
                directory = patch_root / "slides" / slide / "preprocessing"
                payload = read_json(directory / "result.json")
                result = payload["result"]
                dims = result["slide_info"]["dimensions"]
                footprint = result["resolution"]["level0_patch_size"]
                if (
                    payload.get("status") != "success"
                    or len(dims) != 2
                    or any(not isinstance(value, int) or value <= 0 for value in dims)
                    or not np.isfinite(footprint)
                    or footprint <= 0
                ):
                    raise ValueError(f"Invalid preprocessing geometry for {slide}")
                coords = np.load(directory / "coordinates.npy", allow_pickle=False)
                if coords.dtype != np.dtype("int64") or not np.array_equal(
                    coords, file[slide]["coords"][:]
                ):
                    raise ValueError(f"Preprocessing/feature coordinate mismatch: {slide}")
                if (coords[:, 0] >= dims[0]).any() or (coords[:, 1] >= dims[1]).any():
                    raise ValueError(f"Coordinates outside full-slide dimensions: {slide}")
                identity = {
                    "result": hash_file(directory / "result.json"),
                    "coordinates": hash_file(directory / "coordinates.npy"),
                }
                qc = payload.get("files", {}).get("qc")
                if qc and qc.get("thumbnail"):
                    thumbnail = (directory / qc["thumbnail"]).resolve()
                    if not thumbnail.is_relative_to(directory.resolve()):
                        raise ValueError("Preprocessing thumbnail escapes its source directory.")
                    identity["thumbnail"] = hash_file(thumbnail)
                sources[slide] = identity
    return {
        "requested": True,
        "sources": sources,
        "limit_per_seed": config.heatmap_limit,
        "max_size": config.heatmap_max_size,
    }


def evaluate_experiment(*, output, features=None, patch_root=None, resume=False):
    """Evaluate each preselected model once; test metrics never select a model."""
    from lung_wsi_mil.data.mil_dataloader import create_dataset_subset, create_mil_dataloader
    from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
    from lung_wsi_mil.evaluation.metrics import (
        patient_bootstrap_auc,
        prediction_curves,
        summarize_seed_metrics,
    )
    from lung_wsi_mil.experiments.artifacts import load_checkpoint
    from lung_wsi_mil.mil.amil import AMIL
    from lung_wsi_mil.mil.evaluation import evaluate_model
    from lung_wsi_mil.visualization.attention import export_attention, render_from_preprocessing
    from lung_wsi_mil.visualization.figures import plot_repeated_roc, plot_seed_metrics

    output, features, config, manifest, training, partitions, feature_info = load_frozen_run(
        output, features
    )
    test = output / "test"
    if test.exists() and not resume:
        raise FileExistsError(
            "Final-test output exists; --resume only verifies/reuses the same frozen result."
        )
    prior_spec = read_json(test / "rendering.json") if (test / "rendering.json").is_file() else None
    if patch_root is None and prior_spec is not None:
        patch_root = prior_spec.get("source_path")
    patch_root = Path(patch_root).expanduser().resolve() if patch_root else None
    rendering = _rendering_spec(patch_root, partitions, config, features)
    if prior_spec is not None and prior_spec["identity"] != rendering:
        raise ValueError("Final-test rendering sources/settings differ from the saved stage.")
    with run_lock(output):
        if (test / "complete.json").exists():
            verify_stage(test, run_fingerprint=manifest["fingerprint"])
            return read_json(test / "summary.json")
        test.mkdir(parents=True, exist_ok=True)
        atomic_json(
            test / "rendering.json",
            {"identity": rendering, "source_path": str(patch_root) if patch_root else None},
        )
        _status(output, "final_testing", models_selected_before_test=True)
        try:
            results, curves_for_plot, predictions_all = [], [], []
            dataset = MILHDF5Dataset(features)
            for selection in training["seeds"]:
                seed, fold = selection["seed"], selection["selected_fold"]
                directory = test / f"seed_{seed}"
                if (directory / "complete.json").exists():
                    verify_stage(directory, run_fingerprint=manifest["fingerprint"])
                    result = read_json(directory / "summary.json")
                    records = pd.read_csv(directory / "predictions.csv").to_dict("records")
                else:
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir()
                    checkpoint = output / "training" / selection["checkpoint"]
                    if not checkpoint.resolve().is_relative_to((output / "training").resolve()):
                        raise ValueError("Selected checkpoint escapes the training directory.")
                    if hash_file(checkpoint) != selection["checkpoint_sha256"]:
                        raise ValueError(
                            "Selected checkpoint checksum differs from the frozen validation selection."
                        )
                    state = load_checkpoint(
                        checkpoint, run_fingerprint=manifest["fingerprint"], seed=seed, fold=fold
                    )
                    model = AMIL(attention_dim=config.attention_dim, threshold=config.threshold).to(
                        config.device
                    )
                    model.load_state_dict(state["model_state"], strict=True)
                    frame = partitions[seed]["test"]
                    loader = create_mil_dataloader(
                        create_dataset_subset(dataset, frame),
                        config.batch_size,
                        num_workers=config.num_workers,
                        view_indices=(0,),
                    )
                    provenance = prediction_metadata(
                        manifest, seed, fold, selection["checkpoint_sha256"]
                    )
                    heatmap_ids = set(_heatmap_slides(frame, config.heatmap_limit))
                    rendered = []

                    def attention_callback(record, coordinates, weights):
                        attention_csv = export_attention(
                            directory / "attention", record, coordinates, weights, provenance
                        )
                        if patch_root is not None and record["slide_id"] in heatmap_ids:
                            title = record["slide_id"]
                            if not feature_info["official_provenance_verified"]:
                                title = "SYNTHETIC / UNVERIFIED · " + title
                            rendered.append(
                                render_from_preprocessing(
                                    attention_csv,
                                    patch_root / "slides" / record["slide_id"] / "preprocessing",
                                    directory / "heatmaps" / record["slide_id"],
                                    max_size=config.heatmap_max_size,
                                    title=title,
                                )
                            )

                    evaluated = evaluate_model(
                        model,
                        loader,
                        threshold=config.threshold,
                        attention_callback=attention_callback,
                    )
                    records = [{**provenance, **record} for record in evaluated["predictions"]]
                    if set(row["slide_id"] for row in records) != set(frame.slide_id):
                        raise ValueError(
                            "Final-test prediction coverage differs from its frozen holdout."
                        )
                    labels, probabilities = (
                        [row["y_true"] for row in records],
                        [row["probability_tumor"] for row in records],
                    )
                    metrics = evaluated["metrics"]
                    metrics["n_patients"] = len({row["patient_id"] for row in records})
                    uncertainty = patient_bootstrap_auc(
                        labels,
                        probabilities,
                        [row["patient_id"] for row in records],
                        resamples=config.bootstrap_resamples,
                        seed=seed,
                    )
                    atomic_csv(directory / "predictions.csv", records)
                    atomic_json(directory / "metrics.json", metrics)
                    atomic_json(directory / "curves.json", prediction_curves(labels, probabilities))
                    atomic_json(directory / "patient_bootstrap.json", uncertainty)
                    label = f"Seed {seed} · fold {fold} selected on validation"
                    if not feature_info["official_provenance_verified"]:
                        label = "SYNTHETIC / UNVERIFIED · " + label
                    plot_seed_metrics(records, metrics, directory / "metrics", title=label)
                    result = {
                        "seed": seed,
                        "selected_fold": fold,
                        "selected_epoch": selection["selected_epoch"],
                        "checkpoint_sha256": selection["checkpoint_sha256"],
                        "metrics": metrics,
                        "patient_bootstrap": uncertainty,
                        "test_loss": evaluated["loss"],
                        "attention_slides": len(records),
                        "heatmaps": len(rendered),
                        "heatmap_status": "generated" if patch_root else "not_requested",
                        "input_scope": feature_info["input_scope"],
                        "test_used_for_selection": False,
                    }
                    atomic_json(directory / "summary.json", result)
                    seal_stage(
                        directory, run_fingerprint=manifest["fingerprint"], details={"seed": seed}
                    )
                results.append(result)
                curves_for_plot.append((seed, records, result["metrics"]))
                predictions_all.extend(records)
                _status(
                    output,
                    "final_testing",
                    completed_seeds=len(results),
                    expected_seeds=len(config.seeds),
                )
            _assert_feature_stable(features, feature_info, hash_contents=True)
            if _rendering_spec(patch_root, partitions, config, features) != rendering:
                raise ValueError("Spatial rendering sources changed during final testing.")
            if code_identity()["file_sha256"] != manifest["identity"]["code_sha256"]:
                raise ValueError("Experiment code changed during final testing.")
            aggregate = summarize_seed_metrics([item["metrics"] for item in results])
            plot_repeated_roc(
                curves_for_plot,
                test / "repeated_split_roc",
                title="Repeated held-out splits"
                if feature_info["official_provenance_verified"]
                else "SYNTHETIC software smoke test",
            )
            atomic_csv(test / "predictions_all_seeds.csv", predictions_all)
            summary = {
                "status": "complete",
                "run_fingerprint": manifest["fingerprint"],
                "input_scope": feature_info["input_scope"],
                "verified_reference_cohort": manifest["verified_reference_cohort"],
                "completed_training_fits": training["completed_fits"],
                "completed_test_seeds": len(results),
                "reference_five_seeds": set(config.seeds) == {2, 3, 5, 7, 11},
                "seeds": results,
                "aggregate": aggregate,
                "attention_slides": sum(item["attention_slides"] for item in results),
                "heatmaps": sum(item["heatmaps"] for item in results),
                "heatmap_status": "generated" if patch_root else "not_requested",
                "selection_policy": manifest["identity"]["protocol"],
                "pooled_predictions_note": "Combined CSV is for traceability only; repeated slides across seeds are not independent samples.",
            }
            atomic_json(test / "summary.json", summary)
            lines = [
                "# Final held-out AMIL results",
                "",
                f"Input scope: **{feature_info['input_scope']}**.",
                "",
                "Five-fold checkpoint/model selection used validation only. Test used Original views only.",
                "",
                "| Seed | Selected fold | Epoch | Slides | Patients | AUROC | AP | Accuracy |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for item in results:
                metric = item["metrics"]
                lines.append(
                    f"| {item['seed']} | {item['selected_fold']} | {item['selected_epoch']} | {metric['n_slides']} | "
                    f"{metric['n_patients']} | {metric['auroc']:.6f} | {metric['average_precision']:.6f} | {metric['accuracy']:.6f} |"
                )
            lines.extend(
                [
                    "",
                    "Mean/sample SD are descriptive across repeated splits; no pooled independent-cohort AUC is claimed.",
                    "",
                    "Patient-cluster bootstrap intervals, exact predictions and curves are saved per seed.",
                    "Raw attention is saved for every test slide; overlays require the preprocessing geometry/thumbnail root.",
                    "",
                ]
            )
            (test / "report.md").write_text("\n".join(lines), encoding="utf-8")
            seal_stage(
                test,
                run_fingerprint=manifest["fingerprint"],
                details={"completed_seeds": len(results)},
            )
            _status(
                output,
                "complete",
                completed_fits=training["completed_fits"],
                completed_test_seeds=len(results),
            )
            return summary
        except (Exception, KeyboardInterrupt) as error:
            _status(output, "test_interrupted", error=f"{type(error).__name__}: {error}")
            raise
