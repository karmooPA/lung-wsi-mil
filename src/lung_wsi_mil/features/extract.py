#!/usr/bin/env python3
"""Stream verified saved patch views into a genuine KimiaNet MIL feature cache."""

import argparse
import json
from pathlib import Path

from lung_wsi_mil.features.hdf5_pipeline import extract_feature_cache
from lung_wsi_mil.paths import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-root",
        type=Path,
        required=True,
        help="Augmentation output root containing slides/<slide_id>/augmentation/patch_views.h5",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="CSV containing unique slide_id values; omit for all inventory slides",
    )
    parser.add_argument(
        "--inventory", type=Path, default=PROJECT_ROOT / "metadata/dataset_inventory.csv"
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Official trained KimiaNet state dictionary"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New combined MIL HDF5 feature cache"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a fully validated completed cache with identical provenance",
    )
    args = parser.parse_args(argv)
    try:
        result = extract_feature_cache(**vars(args))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Feature extraction failed: {type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
