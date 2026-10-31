#!/usr/bin/env python3
"""Generate the reference patient splits into a new output directory."""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from lung_wsi_mil.data.split_storage import save_all_splits
from lung_wsi_mil.paths import PROJECT_ROOT as PROJECT

METADATA_PATH = PROJECT / "metadata" / "dataset_inventory.csv"
OUTPUT_DIR = PROJECT.parent / f"{PROJECT.name}-outputs" / "splits"


def generate_splits(metadata_path, output_dir):
    """Stage all CSVs before publishing, and refuse to replace existing output."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}. Choose a new --output-dir.")
    metadata = pd.read_csv(metadata_path)
    required = {"slide_id", "patient_id", "label"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if metadata.empty or metadata[list(required)].isna().any().any():
        raise ValueError("Metadata must contain nonempty slide_id, patient_id and label values.")
    for column in required:
        if metadata[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Metadata contains empty {column} values.")
    if metadata["slide_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate slide_id values.")
    if metadata["label"].nunique() != 2:
        raise ValueError("Binary MIL metadata must contain exactly two labels.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
    ) as temporary:
        staged = Path(temporary) / "splits"
        save_all_splits(metadata=metadata, output_dir=staged)
        # Recheck after generation so an existing result is never silently reused.
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output appeared during generation: {output_dir}")
        staged.rename(output_dir)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Input CSV containing slide_id, patient_id and label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="New directory for the five seeds and their CV folds; must not exist.",
    )
    args = parser.parse_args(argv)
    try:
        output = generate_splits(args.metadata, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"All patient splits saved successfully: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
