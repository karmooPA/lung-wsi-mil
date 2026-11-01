"""Train five folds × five seeds, then evaluate frozen models and export attention."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from lung_wsi_mil.experiments.artifacts import hash_file, read_json
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.inputs import load_metadata, validate_features, validate_partitions
from lung_wsi_mil.experiments.pipeline import evaluate_experiment, train_experiment
from lung_wsi_mil.paths import PROJECT_ROOT


def _config(args):
    if args.config:
        config = ExperimentConfig.from_json(args.config)
    elif getattr(args, "resume", False) and (args.output / "run.json").is_file():
        config = ExperimentConfig.from_dict(read_json(args.output / "run.json")["config"])
    else:
        config = ExperimentConfig()
    overrides = {
        name: getattr(args, name)
        for name in (
            "epochs",
            "patience",
            "batch_size",
            "device",
            "seeds",
            "bootstrap_resamples",
            "heatmap_limit",
            "allow_unverified_features",
        )
        if getattr(args, name, None) is not None
    }
    return replace(config, **overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="Exercise all 25 fits on labelled synthetic inputs; not real-data acceptance."
    )
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--patients", type=int, default=30)
    smoke.add_argument("--epochs", type=int, default=2)
    for name in ("validate", "train", "run"):
        command = commands.add_parser(name)
        command.add_argument("--features", type=Path, required=True)
        command.add_argument(
            "--inventory", type=Path, default=PROJECT_ROOT / "metadata/dataset_inventory.csv"
        )
        command.add_argument("--splits", type=Path, default=PROJECT_ROOT / "splits")
        command.add_argument("--config", type=Path)
        command.add_argument("--epochs", type=int)
        command.add_argument("--patience", type=int)
        command.add_argument("--batch-size", type=int)
        command.add_argument("--device")
        command.add_argument("--seeds", type=int, nargs="+")
        command.add_argument("--bootstrap-resamples", type=int)
        command.add_argument("--heatmap-limit", type=int)
        command.add_argument(
            "--allow-unverified-features",
            action="store_true",
            default=None,
            help="Explicitly label a synthetic/legacy cache as unverified; never report real-data acceptance.",
        )
        if name != "validate":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
        if name == "run":
            command.add_argument(
                "--patch-root",
                type=Path,
                help="Saved per-slide preprocessing geometry and QC thumbnails for spatial heatmaps.",
            )
    command = commands.add_parser("test")
    command.add_argument("--run-dir", dest="output", type=Path, required=True)
    command.add_argument(
        "--features", type=Path, help="Optional moved copy with exactly the training cache hash."
    )
    command.add_argument("--patch-root", type=Path)
    command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            from lung_wsi_mil.experiments.synthetic import make_synthetic_inputs

            root = args.output.expanduser().resolve()
            if root.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(root):
                raise ValueError("Software-smoke output must be outside the source repository.")
            if root.exists():
                raise FileExistsError(f"Choose a new software-smoke output folder: {root}")
            config = ExperimentConfig(
                epochs=args.epochs,
                patience=2,
                attention_dim=8,
                batch_size=4,
                bootstrap_resamples=50,
                heatmap_limit=2,
                heatmap_max_size=256,
                allow_unverified_features=True,
            )
            inputs = make_synthetic_inputs(root / "inputs", patients=args.patients)
            train_experiment(
                features=inputs["features"],
                inventory=inputs["inventory"],
                splits=inputs["splits"],
                output=root / "experiment",
                config=config,
            )
            report = evaluate_experiment(
                output=root / "experiment", patch_root=inputs["patch_root"]
            )
        elif args.command == "test":
            report = evaluate_experiment(
                output=args.output,
                features=args.features,
                patch_root=args.patch_root,
                resume=args.resume,
            )
        else:
            config = _config(args)
            if args.command == "validate":
                metadata = load_metadata(args.inventory)
                validate_partitions(metadata, args.splits, config)
                report = {
                    "status": "validated",
                    **validate_features(
                        args.features,
                        metadata,
                        inventory_sha256=hash_file(args.inventory),
                        allow_unverified=config.allow_unverified_features,
                    ),
                    "requested_seeds": list(config.seeds),
                    "folds_per_seed": config.folds,
                }
            else:
                report = train_experiment(
                    features=args.features,
                    inventory=args.inventory,
                    splits=args.splits,
                    output=args.output,
                    config=config,
                    resume=args.resume,
                )
                if args.command == "run":
                    report = evaluate_experiment(
                        output=args.output,
                        features=args.features,
                        patch_root=args.patch_root,
                        resume=args.resume,
                    )
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted. Epoch checkpoints are retained; use the same inputs/config with --resume.\n",
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Experiment failed: {error}\n")
    compact = {key: value for key, value in report.items() if key not in {"seeds"}}
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
