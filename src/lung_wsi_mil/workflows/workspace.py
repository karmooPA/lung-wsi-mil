"""Run the verified WSI pipeline in a structured external workspace."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lung_wsi_mil.data.generate_splits import generate_splits
from lung_wsi_mil.experiments.artifacts import atomic_json, hash_file, read_json, run_lock
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.inputs import load_metadata, validate_features, validate_partitions
from lung_wsi_mil.experiments.pipeline import _rendering_spec
from lung_wsi_mil.features.kimianet import OFFICIAL_WEIGHTS_SHA256
from lung_wsi_mil.paths import PROJECT_ROOT
from lung_wsi_mil.workflows import dataset, pilot


def workspace_path(root, *parts):
    path = root.joinpath(*parts)
    if path.resolve() != path:
        raise ValueError(f"Workspace path is redirected: {path}")
    return path


def check_sources(args, paths, metadata):
    if args.start_at == "raw":
        selected = dataset._select_all_present(
            pilot._load_inventory(args.inventory), *pilot._index_local_slides(paths["raw"])
        )
        if {row["slide_id"] for row in selected} != set(metadata.slide_id):
            raise ValueError("Raw WSI coverage differs from the inventory.")
        if any(not Path(row["resolved_path"]).is_relative_to(paths["raw"]) for row in selected):
            raise ValueError("A raw slide resolves outside input/wsi.")
    if args.start_at in ("raw", "patches"):
        if hash_file(paths["weights"]) != OFFICIAL_WEIGHTS_SHA256:
            raise ValueError("KimiaNet weights do not match the pinned official checkpoint.")
    if args.start_at == "patches":
        for slide in metadata.slide_id:
            source = paths["patches"] / "slides" / slide / "augmentation/patch_views.h5"
            if not source.is_file():
                raise FileNotFoundError(source)


def feature_preflight(args, paths, metadata, partitions, config):
    features = validate_features(
        paths["features"], metadata, inventory_sha256=metadata.attrs["source_sha256"]
    )
    if not args.no_heatmaps:
        rendering = _rendering_spec(paths["patches"], partitions, config, paths["features"])
        missing = [
            slide for slide, source in rendering["sources"].items() if "thumbnail" not in source
        ]
        if missing:
            raise ValueError(f"Spatial overlays need preprocessing thumbnails: {missing}")
    return features


def call_stage(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True)


def run(args):
    root = args.data_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(root):
        raise ValueError("Data must be an existing directory separate from the source checkout.")
    name = args.run_name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or re.fullmatch(
        r"CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]", name, re.IGNORECASE
    ):
        raise ValueError("Run name must use 1–64 safe letters, digits, underscores or hyphens.")
    run_root = workspace_path(root, "output", name)
    generated = workspace_path(root, "derived", "runs", name)
    if args.start_at == "features":
        feature_path = workspace_path(root, "derived", "features", "kimianet.h5")
    else:
        feature_path = workspace_path(generated, "kimianet.h5")
    paths = {
        "raw": workspace_path(root, "input", "wsi"),
        "weights": workspace_path(root, "input", "weights", "KimiaNetPyTorchWeights.pth"),
        "patches": (
            workspace_path(generated, "patches")
            if args.start_at == "raw"
            else workspace_path(root, "derived", "patches")
        ),
        "features": feature_path,
        "experiment": workspace_path(run_root, "experiment"),
    }
    args.inventory = args.inventory.expanduser().resolve(strict=True)
    args.config = args.config.expanduser().resolve(strict=True)
    config = ExperimentConfig.from_json(args.config)
    config = replace(config, device=args.device or config.device)
    if config.allow_unverified_features or config.device not in ("cpu", "cuda"):
        raise ValueError("Workspace runs require verified features and a cpu or cuda device.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    metadata = load_metadata(args.inventory)
    check_sources(args, paths, metadata)
    split_root = (
        workspace_path(run_root, "splits")
        if args.generate_splits
        else args.splits.expanduser().resolve(strict=True)
    )
    if args.check_only and args.generate_splits:
        with TemporaryDirectory() as temporary:
            staged = generate_splits(args.inventory, Path(temporary) / "splits")
            partitions, split_hashes = validate_partitions(metadata, staged, config)
    elif not args.generate_splits:
        partitions, split_hashes = validate_partitions(metadata, split_root, config)
    else:
        partitions, split_hashes = None, None

    feature_info = (
        feature_preflight(args, paths, metadata, partitions, config)
        if args.start_at == "features" and partitions is not None
        else None
    )
    if args.check_only:
        return {
            "status": "validated",
            "start_at": args.start_at,
            "slides": len(metadata),
            "patients": int(metadata.patient_id.nunique()),
            "seeds": list(config.seeds),
            "folds": config.folds,
            "features": feature_info,
            "paths": {key: str(value) for key, value in paths.items()},
        }

    identity = {
        "schema_version": 1,
        "runner_sha256": hash_file(__file__),
        "paths": {key: str(value) for key, value in paths.items()},
        "start_at": args.start_at,
        "inventory": str(args.inventory),
        "inventory_sha256": metadata.attrs["source_sha256"],
        "splits": str(split_root),
        "split_sha256": split_hashes,
        "config": str(args.config),
        "config_sha256": hash_file(args.config),
        "device": config.device,
        "heatmaps": not args.no_heatmaps,
        "workers": args.workers,
        "feature_batch_size": args.feature_batch_size,
        "input_feature_sha256": feature_info["sha256"] if feature_info else None,
    }
    manifest_path = workspace_path(run_root, "workspace.json")
    status_path = workspace_path(run_root, "workspace-status.json")
    if not args.resume:
        if args.start_at != "features" and generated.exists():
            raise FileExistsError(f"Run-scoped derived data already exists: {generated}")
        run_root.mkdir(parents=True, exist_ok=False)

    with run_lock(run_root):
        if args.generate_splits:
            if not args.resume:
                generate_splits(args.inventory, split_root)
            partitions, identity["split_sha256"] = validate_partitions(metadata, split_root, config)
        if args.start_at == "features" and feature_info is None:
            feature_info = feature_preflight(args, paths, metadata, partitions, config)
            identity["input_feature_sha256"] = feature_info["sha256"]
        if args.resume:
            if read_json(manifest_path) != identity:
                raise ValueError("Workspace inputs or settings differ from the recorded run.")
        else:
            atomic_json(manifest_path, identity)

        complete = workspace_path(paths["experiment"], "test", "complete.json").is_file()
        phase = "splits"

        def status(state, **extra):
            atomic_json(
                status_path,
                {
                    "status": state,
                    "phase": phase,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    **extra,
                },
            )

        try:
            status("running")
            if args.start_at == "raw" and not complete:
                phase = "preprocess"
                status("running")
                call_stage(
                    "lung_wsi_mil.workflows.dataset",
                    paths["raw"],
                    paths["patches"],
                    "--inventory",
                    args.inventory,
                    "--verify-source-md5",
                    "--workers",
                    args.workers,
                )
                phase = "verify-patches"
                status("running")
                call_stage(
                    "lung_wsi_mil.workflows.verify_dataset",
                    paths["patches"],
                    "--verify-source-md5",
                    "--workers",
                    args.workers,
                )
            if args.start_at != "features" and not complete:
                phase = "extract-features"
                status("running")
                call_stage(
                    "lung_wsi_mil.features.extract",
                    "--patch-root",
                    paths["patches"],
                    "--inventory",
                    args.inventory,
                    "--weights",
                    paths["weights"],
                    "--output",
                    paths["features"],
                    "--batch-size",
                    args.feature_batch_size,
                    "--device",
                    config.device,
                    *(("--resume",) if args.resume else ()),
                )
            phase = "validate-features"
            status("running")
            if feature_info is None:
                feature_info = feature_preflight(args, paths, metadata, partitions, config)
            phase = "train-and-test"
            status("running")
            experiment_args = [
                "run",
                "--features",
                paths["features"],
                "--inventory",
                args.inventory,
                "--splits",
                split_root,
                "--config",
                args.config,
                "--device",
                config.device,
                "--output",
                paths["experiment"],
            ]
            if not args.no_heatmaps:
                experiment_args += ["--patch-root", paths["patches"]]
            if args.resume:
                experiment_args.append("--resume")
            call_stage("lung_wsi_mil.experiments.cli", *experiment_args)
            summary_path = paths["experiment"] / "test" / "summary.json"
            report = read_json(summary_path)
            if report["status"] != "complete":
                raise ValueError("Experiment did not produce a complete test report.")
            phase = "complete"
            result = {
                "output": str(run_root),
                "summary": str(summary_path),
                "training_fits": report["completed_training_fits"],
                "test_seeds": report["completed_test_seeds"],
                "feature_sha256": feature_info["sha256"],
            }
            status("complete", **result)
            return {"status": "complete", **result}
        except (Exception, KeyboardInterrupt) as error:
            status(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--start-at", choices=("raw", "patches", "features"), default="features")
    parser.add_argument(
        "--inventory", type=Path, default=PROJECT_ROOT / "metadata/dataset_inventory.csv"
    )
    splits = parser.add_mutually_exclusive_group()
    splits.add_argument("--splits", type=Path, default=PROJECT_ROOT / "splits")
    splits.add_argument("--generate-splits", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/amil-reference.json")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--no-heatmaps", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument(
        "--check-only", action="store_true", help="Validate inputs without writing a run."
    )
    args = parser.parse_args(argv)
    if min(args.workers, args.feature_batch_size) < 1:
        parser.error("--workers and --feature-batch-size must be positive.")
    try:
        report = run(args)
    except KeyboardInterrupt:
        parser.exit(130, "Workspace run interrupted. Use the same arguments with --resume.\n")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"Workspace pipeline failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
