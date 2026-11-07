"""Deliver every frozen fold's test results and all spatial overlays separately."""

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from matplotlib import colormaps
from PIL import Image

from lung_wsi_mil.data.mil_dataloader import create_dataset_subset, create_mil_dataloader
from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
from lung_wsi_mil.evaluation.metrics import patient_bootstrap_auc, prediction_curves
from lung_wsi_mil.experiments.artifacts import (
    atomic_csv,
    atomic_json,
    fingerprint,
    hash_file,
    load_checkpoint,
    read_json,
    run_lock,
    seal_stage,
    verify_stage,
)
from lung_wsi_mil.experiments.fold import prediction_metadata
from lung_wsi_mil.experiments.pipeline import load_frozen_run
from lung_wsi_mil.mil.amil import AMIL
from lung_wsi_mil.mil.evaluation import evaluate_model
from lung_wsi_mil.paths import PROJECT_ROOT
from lung_wsi_mil.visualization.attention import (
    export_attention,
    rasterize_attention,
    read_attention,
)
from lung_wsi_mil.visualization.figures import plot_seed_metrics

REFERENCE_SEEDS = (2, 3, 5, 7, 11)
IMAGE_NAMES = ("overlay.png", "heatmap.png", "coverage.png")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(UTC).isoformat()


def _geometry(patch_root, slide):
    root = patch_root.resolve()
    directory = (root / "slides" / slide / "preprocessing").resolve()
    _require(directory.is_relative_to(root), f"{slide}: preprocessing path escapes its root")
    result_path = directory / "result.json"
    coords_path = directory / "coordinates.npy"
    payload = read_json(result_path)
    result = payload["result"]
    dims = result["slide_info"]["dimensions"]
    footprint = result["resolution"]["level0_patch_size"]
    _require(
        payload["status"] == "success"
        and len(dims) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dims
        )
        and math.isfinite(footprint)
        and footprint > 0,
        f"{slide}: invalid saved preprocessing geometry",
    )
    thumbnail = (directory / payload["files"]["qc"]["thumbnail"]).resolve()
    _require(
        thumbnail.is_relative_to(directory) and thumbnail.is_file(),
        f"{slide}: required source thumbnail is missing or outside preprocessing",
    )
    return dims, footprint, result["valid_patch_count"], result_path, coords_path, thumbnail


def _geometry_inventory(patch_root, partitions, features):
    """Verify every unique holdout geometry against the exact feature coordinates."""
    slides = sorted({slide for frames in partitions.values() for slide in frames["test"].slide_id})
    identity = {}
    with h5py.File(features, "r") as cache:
        for slide in slides:
            dims, footprint, patches, result, coordinates, thumbnail = _geometry(patch_root, slide)
            coords = np.load(coordinates, allow_pickle=False)
            _require(
                coords.dtype == np.dtype("int64")
                and coords.shape == (patches, 2)
                and np.array_equal(coords, cache[slide]["coords"][:])
                and (coords >= 0).all()
                and (coords[:, 0] < dims[0]).all()
                and (coords[:, 1] < dims[1]).all(),
                f"{slide}: saved preprocessing coordinates differ from features",
            )
            with Image.open(thumbnail) as image:
                ratio = dims[0] / dims[1]
                _require(
                    abs(image.width - image.height * ratio) <= max(2, image.width * 0.01),
                    f"{slide}: thumbnail aspect ratio differs from slide geometry",
                )
            identity[slide] = {
                "result_sha256": hash_file(result),
                "coordinates_sha256": hash_file(coordinates),
                "thumbnail_sha256": hash_file(thumbnail),
                "slide_dimensions_level0": dims,
                "patch_size_level0": footprint,
                "patches": patches,
            }
    return identity


def _render_slide(slide, attention_path, patch_root, target, geometry, max_size):
    """Use the standard rank/overlap raster and RGBA overlay without figure/NPY duplication."""
    observed_slide, coords, weights = read_attention(attention_path)
    dims, footprint, patches, result, coordinates, thumbnail = _geometry(patch_root, slide)
    _require(
        observed_slide == slide
        and len(weights) == patches == geometry["patches"]
        and hash_file(result) == geometry["result_sha256"]
        and hash_file(coordinates) == geometry["coordinates_sha256"]
        and hash_file(thumbnail) == geometry["thumbnail_sha256"]
        and np.array_equal(coords, np.load(coordinates, allow_pickle=False)),
        f"{slide}: attention and saved geometry are not aligned",
    )
    scale = min(1, max_size / max(dims))
    size = tuple(max(1, int(round(value * scale))) for value in dims)
    _, display, coverage = rasterize_attention(
        coords, weights, slide_dimensions=dims, patch_size_level0=footprint, size=size
    )
    covered = coverage > 0
    rgba = (colormaps["viridis"](np.nan_to_num(display, nan=0.5)) * 255).round().astype(np.uint8)
    rgba[:, :, 3] = np.where(covered, 255, 0)
    target.mkdir()
    Image.fromarray(rgba).save(target / "heatmap.png")
    Image.fromarray(covered.astype(np.uint8) * 255).save(target / "coverage.png")
    rgba[:, :, 3] = np.where(covered, round(0.45 * 255), 0)
    with Image.open(thumbnail) as image:
        background = image.convert("RGB").resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    Image.alpha_composite(background, Image.fromarray(rgba)).convert("RGB").save(
        target / "overlay.png"
    )
    metadata = {
        "slide_id": slide,
        "raster_size": list(size),
        "covered_pixels": int(covered.sum()),
        "view": "Original",
        "alpha": 0.45,
        "overlay_generated": True,
        **geometry,
        "source_attention_sha256": hash_file(attention_path),
        "display": "within-slide average-rank percentiles; overlapping footprints averaged",
        "unsampled_regions": "transparent; no extrapolation or smoothing",
        "interpretation": "Model attention is not a tumor mask or localization ground truth.",
    }
    atomic_json(target / "heatmap.json", metadata)
    return metadata


def _evaluate_pair(
    *,
    pair,
    seed,
    fold,
    train_source,
    selection,
    dataset,
    frame,
    config,
    manifest,
    patch_root,
    geometry,
    render_workers,
    max_size,
    delivery_id,
):
    from lung_wsi_mil.workflows.fold_report import write_fold_report

    run_id = manifest["fingerprint"]
    verify_stage(train_source, run_fingerprint=run_id)
    train_summary = read_json(train_source / "summary.json")
    saved_fold = next(row for row in selection["folds"] if row["fold"] == fold)
    _require(
        train_summary == saved_fold
        and train_summary["seed"] == seed
        and train_summary["fold"] == fold
        and train_summary["test_used_for_selection"] is False,
        f"Seed {seed} fold {fold}: fold summary differs from the frozen selection inputs",
    )
    shutil.copytree(train_source, pair / "train")
    verify_stage(pair / "train", run_fingerprint=run_id)
    checkpoint = pair / "train/best.pt"
    checkpoint_sha = hash_file(checkpoint)
    state = load_checkpoint(checkpoint, run_fingerprint=run_id, seed=seed, fold=fold)
    best_epoch = train_summary["best_epoch"]
    primary = selection["selected_fold"] == fold
    _require(
        checkpoint_sha == train_summary["checkpoint_sha256"]
        and state["epoch"] == best_epoch
        and fingerprint(state["config"]) == manifest["config_hash"]
        and (
            not primary
            or (
                selection["selected_epoch"] == best_epoch
                and selection["checkpoint_sha256"] == checkpoint_sha
            )
        ),
        f"Seed {seed} fold {fold}: invalid frozen checkpoint identity",
    )
    test = pair / "test"
    test.mkdir()
    provenance = {
        **prediction_metadata(manifest, seed, fold, checkpoint_sha),
        "delivery_id": delivery_id,
        "validation_selected_for_seed": primary,
    }
    model = AMIL(attention_dim=config.attention_dim, threshold=config.threshold).to(config.device)
    model.load_state_dict(state["model_state"], strict=True)
    model.requires_grad_(False)
    loader = create_mil_dataloader(
        create_dataset_subset(dataset, frame),
        config.batch_size,
        num_workers=config.num_workers,
        view_indices=(0,),
    )

    def attention_callback(record, coordinates, weights):
        export_attention(test / "attention", record, coordinates, weights, provenance)

    evaluated = evaluate_model(
        model, loader, threshold=config.threshold, attention_callback=attention_callback
    )
    records = [{**provenance, **row} for row in evaluated["predictions"]]
    _require(
        [(r["slide_id"], r["patient_id"], r["y_true"], r["view"]) for r in records]
        == [(r.slide_id, r.patient_id, r.label, "Original") for r in frame.itertuples()],
        f"Seed {seed} fold {fold}: predictions differ from the frozen Original holdout",
    )
    labels = [r["y_true"] for r in records]
    probabilities = [r["probability_tumor"] for r in records]
    metrics = {**evaluated["metrics"], "n_patients": int(frame.patient_id.nunique())}
    uncertainty = patient_bootstrap_auc(
        labels,
        probabilities,
        [r["patient_id"] for r in records],
        resamples=config.bootstrap_resamples,
        seed=seed,
    )
    atomic_csv(test / "predictions.csv", records)
    atomic_json(test / "metrics.json", metrics)
    atomic_json(test / "patient_bootstrap.json", uncertainty)
    atomic_json(test / "curves.json", prediction_curves(labels, probabilities))
    plot_seed_metrics(
        records, metrics, test / "metrics", title=f"Seed {seed} · fold {fold} · held-out test"
    )
    heatmaps = test / "heatmaps"
    heatmaps.mkdir()

    def render(record):
        slide = record["slide_id"]
        return _render_slide(
            slide,
            test / "attention" / f"{slide}.csv",
            patch_root,
            heatmaps / slide,
            geometry[slide],
            max_size,
        )

    index = []
    examples = {}
    with ThreadPoolExecutor(max_workers=render_workers) as pool:
        for number, (record, metadata) in enumerate(
            zip(records, pool.map(render, records), strict=True), 1
        ):
            slide = record["slide_id"]
            category = {
                (0, 0): "true_negative",
                (0, 1): "false_positive",
                (1, 0): "false_negative",
                (1, 1): "true_positive",
            }[(record["y_true"], record["y_pred"])]
            paths = {
                name.removesuffix(".png"): f"test/heatmaps/{slide}/{name}" for name in IMAGE_NAMES
            }
            examples.setdefault(category, paths["overlay"])
            index.append(
                {
                    "slide_id": slide,
                    "patient_id": record["patient_id"],
                    "y_true": record["y_true"],
                    "y_pred": record["y_pred"],
                    "probability_tumor": record["probability_tumor"],
                    "category": category,
                    "patch_count": metadata["patches"],
                    "covered_pixels": metadata["covered_pixels"],
                    **paths,
                    "metadata": f"test/heatmaps/{slide}/heatmap.json",
                }
            )
            if number % 25 == 0 or number == len(records):
                print(f"  seed {seed} fold {fold}: {number}/{len(records)} overlays", flush=True)
    atomic_csv(heatmaps / "index.csv", index)
    history = read_json(pair / "train/history.json")
    _require(
        len(history) == train_summary["epochs_run"] and 1 <= best_epoch <= len(history),
        f"Seed {seed} fold {fold}: incomplete training history",
    )
    validation = train_summary["validation"]
    write_fold_report(
        pair,
        seed=seed,
        fold=fold,
        validation=validation,
        test=metrics,
        uncertainty=uncertainty,
        history=history,
        predictions=records,
        selected_epoch=best_epoch,
        primary_selected=primary,
        checkpoint_sha256=checkpoint_sha,
        cv_loss=validation["loss"],
        test_loss=evaluated["loss"],
        heatmap_index="test/heatmaps/index.csv",
        metrics_image="test/metrics.png",
        error_examples=examples,
        run_fingerprint=run_id,
    )
    result = {
        "seed": seed,
        "fold": fold,
        "selected_epoch": best_epoch,
        "validation_selected_for_seed": primary,
        "checkpoint_sha256": checkpoint_sha,
        "metrics": metrics,
        "validation_metrics": validation,
        "test_loss": evaluated["loss"],
        "patient_bootstrap": uncertainty,
        "attention_slides": len(records),
        "heatmaps": len(index),
        "test_used_for_selection": False,
        "training_run_fingerprint": run_id,
        "delivery_id": delivery_id,
    }
    atomic_json(test / "summary.json", result)
    seal_stage(
        pair,
        run_fingerprint=delivery_id,
        details={"seed": seed, "fold": fold, "heatmaps": len(index), "training_run": run_id},
    )
    verify_stage(pair, run_fingerprint=delivery_id)
    return result


def deliver_folds(run_dir, patch_root, output, *, resume=False, render_workers=4, max_size=1024):
    """Evaluate all 25 checkpoints; publish each verified pair atomically to staging."""
    run_dir, patch_root, output = (
        Path(path).expanduser().resolve() for path in (run_dir, patch_root, output)
    )
    _require(1 <= render_workers <= 16 and 1 <= max_size <= 1024, "Invalid rendering resources")
    for protected in (PROJECT_ROOT.resolve(), run_dir, patch_root):
        _require(
            not output.is_relative_to(protected) and not protected.is_relative_to(output),
            "Delivery staging must be disjoint from the checkout and input artifacts",
        )
    run_dir, features, config, manifest, training, partitions, info = load_frozen_run(run_dir)
    _require(
        not features.is_relative_to(output), "Delivery staging may not contain its feature input"
    )
    _require(
        tuple(config.seeds) == REFERENCE_SEEDS
        and config.folds == 5
        and training["completed_fits"] == 25
        and info["official_provenance_verified"],
        "All 25 reference fits and official feature provenance are required",
    )
    _require(
        [row["seed"] for row in training["seeds"]] == list(REFERENCE_SEEDS),
        "Frozen training summary has unexpected seed order or coverage",
    )
    geometry = _geometry_inventory(patch_root, partitions, features)
    run_id = manifest["fingerprint"]
    training_seal = hash_file(run_dir / "training/complete.json")
    sources = [Path(__file__), Path(__file__).with_name("fold_report.py")]
    identity = {
        "training_run": run_id,
        "training_seal_sha256": training_seal,
        "features_sha256": info["sha256"],
        "max_size": max_size,
        "geometry": geometry,
        "code_sha256": {path.name: hash_file(path) for path in sources},
        "test_scope": "all 25 frozen fold checkpoints, each on its seed's Original holdout",
    }
    delivery_id = fingerprint(identity)
    if output.exists() and not resume:
        raise FileExistsError(f"Delivery staging already exists: {output}; use --resume")
    output.mkdir(parents=True, exist_ok=resume)
    with run_lock(output):
        plan = output / "run.json"
        if plan.exists():
            _require(
                read_json(plan) == {"fingerprint": delivery_id, "identity": identity},
                "Delivery resume refused: frozen training, geometry, code or render size changed",
            )
        elif any(path.name != "run.lock" for path in output.iterdir()):
            raise ValueError(
                "Existing delivery staging is not empty and has no verified run identity"
            )
        else:
            atomic_json(plan, {"fingerprint": delivery_id, "identity": identity})
        dataset = MILHDF5Dataset(features)
        results = []

        def progress(status, **kwargs):
            atomic_json(
                output / "status.json",
                {
                    "status": status,
                    "updated_utc": _now(),
                    "completed_models": len(results),
                    "expected_models": 25,
                    **kwargs,
                },
            )

        try:
            for selection in training["seeds"]:
                seed = selection["seed"]
                frame = partitions[seed]["test"]
                for fold in range(1, 6):
                    name = f"seed_{seed:02d}_fold_{fold:02d}"
                    pair = output / name
                    progress("evaluating", seed=seed, fold=fold)
                    print(f"[{len(results) + 1}/25] {name}", flush=True)
                    if pair.exists():
                        details = verify_stage(pair, run_fingerprint=delivery_id)
                        verify_stage(pair / "train", run_fingerprint=run_id)
                        _require(
                            (details["seed"], details["fold"], details["heatmaps"])
                            == (seed, fold, len(frame)),
                            f"Incorrect sealed pair identity: {name}",
                        )
                        result = read_json(pair / "test/summary.json")
                    else:
                        with TemporaryDirectory(prefix=f".{name}-", dir=output) as temporary:
                            staged = Path(temporary) / name
                            staged.mkdir()
                            result = _evaluate_pair(
                                pair=staged,
                                seed=seed,
                                fold=fold,
                                train_source=run_dir / "training" / f"seed_{seed}" / f"fold_{fold}",
                                selection=selection,
                                dataset=dataset,
                                frame=frame,
                                config=config,
                                manifest=manifest,
                                patch_root=patch_root,
                                geometry=geometry,
                                render_workers=render_workers,
                                max_size=max_size,
                                delivery_id=delivery_id,
                            )
                            staged.rename(pair)
                    results.append(result)
                    progress("evaluating", last_completed=name)
            reloaded = load_frozen_run(run_dir)
            _require(
                reloaded[3]["fingerprint"] == run_id
                and reloaded[6]["sha256"] == info["sha256"]
                and hash_file(run_dir / "training/complete.json") == training_seal
                and _geometry_inventory(patch_root, partitions, features) == geometry
                and {p.name: hash_file(p) for p in sources} == identity["code_sha256"],
                "Training, feature, preprocessing geometry or delivery code changed during export",
            )
            summary = {
                "status": "complete",
                "delivery_id": delivery_id,
                "training_run": run_id,
                "models": len(results),
                "heatmaps": sum(result["heatmaps"] for result in results),
                "test_used_for_selection": False,
                "pairs": results,
                "finished_utc": _now(),
                "interpretation": "Shared per-seed holdouts; no pooled independent-cohort metrics.",
            }
            atomic_json(output / "summary.json", summary)
            progress("complete", heatmaps=summary["heatmaps"])
            return summary
        except BaseException as error:
            progress("failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        report = deliver_folds(
            args.run_dir,
            args.patch_root,
            args.output,
            resume=args.resume,
            render_workers=args.render_workers,
            max_size=args.max_size,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Fold delivery failed: {error}\n")
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
