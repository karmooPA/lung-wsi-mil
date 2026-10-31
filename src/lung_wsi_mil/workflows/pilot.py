#!/usr/bin/env python3
"""Run preprocessing + augmentation on a balanced 14-WSI pilot cohort.

The script reuses the project's existing scientific implementation without
changing its preprocessing or augmentation rules.  It selects seven Tumor and
seven Normal slides that are actually present under a dataset root, runs the
preprocessing pipeline, extracts every valid patch, applies Original/Aug1/Aug2
in bounded chunks, and saves the results losslessly in compressed HDF5 files.

Example:
    python src/lung_wsi_mil/workflows/pilot.py \
        /data/TCGA-LUSC \
        ../lung-wsi-mil-outputs/pilot14 \
        --inventory metadata/dataset_inventory.csv \
        --seed 42
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from lung_wsi_mil.augmentation import transform_batch
from lung_wsi_mil.augmentation.data_augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
)
from lung_wsi_mil.augmentation.storage import (
    REPRESENTATIONS,
    normalized_views_to_uint8,
    uint8_views_to_normalized,
    validate_augmented_hdf5,
)
from lung_wsi_mil.paths import PACKAGE_ROOT, PROJECT_ROOT
from lung_wsi_mil.preprocessing.batch_preprocessing import (
    build_config,
    config_fingerprint,
    save_slide_outputs,
    validate_saved_outputs,
)
from lung_wsi_mil.preprocessing.patch_extraction import extract_patch
from lung_wsi_mil.preprocessing.pipeline import preprocess_slide
from lung_wsi_mil.preprocessing.wsi_reader import open_slide

SUPPORTED_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
PILOT_VERSION = "2.0.0"
HDF5_FORMAT_VERSION = "2.0"
LABELS = ("Tumor", "Normal")
PER_CLASS = 7
NA_VALUES = {"", "NA", "N/A", "NONE", "NULL"}
CODE_INPUTS = (
    "src/lung_wsi_mil/paths.py",
    "src/lung_wsi_mil/workflows/pilot.py",
    "src/lung_wsi_mil/augmentation/data_augmentations.py",
    "src/lung_wsi_mil/augmentation/storage.py",
    "src/lung_wsi_mil/preprocessing/pipeline.py",
    "src/lung_wsi_mil/preprocessing/wsi_reader.py",
    "src/lung_wsi_mil/preprocessing/thumbnail.py",
    "src/lung_wsi_mil/preprocessing/tissue_detection.py",
    "src/lung_wsi_mil/preprocessing/patch_coordinates.py",
    "src/lung_wsi_mil/preprocessing/patch_extraction.py",
    "src/lung_wsi_mil/preprocessing/batch_preprocessing.py",
    "src/lung_wsi_mil/preprocessing/qc.py",
)


def _write_json_atomic(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _hash_file(path, algorithm="sha256", block_size=4 * 1024 * 1024):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_fingerprint(project_root=None):
    digest = hashlib.sha256()
    for relative in CODE_INPUTS:
        path = PACKAGE_ROOT / relative.removeprefix("src/lung_wsi_mil/")
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline code input is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_versions():
    packages = (
        "numpy",
        "Pillow",
        "scikit-image",
        "torch",
        "torchvision",
        "h5py",
        "openslide-python",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _acquire_run_lock(output_root):
    lock_path = Path(output_root) / ".pilot14.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Another pilot run may be active ({lock_path}; {owner or 'unknown owner'}). "
            "Remove the lock only after confirming that no run is active."
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    return lock_path


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _load_inventory(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Inventory not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "patient_id", "label", "relative_path"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError("Inventory is missing required columns: " + ", ".join(sorted(missing)))
    return rows


def _index_local_slides(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    by_name = {}
    by_slide_prefix = {}
    for path in paths:
        by_name.setdefault(path.name, []).append(path)
        slide_prefix = path.name.split(".", 1)[0]
        by_slide_prefix.setdefault(slide_prefix, []).append(path)
    return dataset_root, paths, by_name, by_slide_prefix


def _resolve_inventory_row(row, dataset_root, by_name, by_slide_prefix):
    relative = str(row.get("relative_path") or "").strip()
    if relative:
        exact = (dataset_root / relative).resolve()
        if not exact.is_relative_to(dataset_root):
            raise ValueError(f"Inventory relative_path escapes the dataset root: {relative!r}")
        if exact.is_file():
            return exact, "relative_path"

    file_name = str(row.get("file_name") or "").strip()
    if not file_name and relative:
        file_name = Path(relative).name
    name_matches = by_name.get(file_name, []) if file_name else []
    if len(name_matches) == 1:
        return name_matches[0].resolve(), "file_name"

    slide_id = str(row.get("slide_id") or "").strip()
    prefix_matches = by_slide_prefix.get(slide_id, []) if slide_id else []
    if len(prefix_matches) == 1:
        return prefix_matches[0].resolve(), "slide_id_prefix"

    return None, None


def _eligible_present_rows(inventory_rows, dataset_root, by_name, by_slide_prefix):
    present = []
    for row in inventory_rows:
        label = str(row.get("label") or "").strip()
        if label not in LABELS:
            continue
        if _parse_bool(row.get("duplicate")) is True:
            continue
        label_valid = _parse_bool(row.get("label_valid"))
        if label_valid is False:
            continue
        patient_id = str(row.get("patient_id") or "").strip()
        if not patient_id or patient_id.upper() in NA_VALUES:
            continue

        path, resolution_method = _resolve_inventory_row(
            row,
            dataset_root,
            by_name,
            by_slide_prefix,
        )
        if path is None:
            continue

        item = dict(row)
        item["resolved_path"] = str(path)
        item["path_resolution_method"] = resolution_method
        present.append(item)
    return present


def _select_balanced(rows, seed):
    rng = np.random.default_rng(seed)
    selected = []
    used_patients = set()

    # Alternating labels prevents one class from consuming shared TCGA patients
    # that might also have an opposite-label slide.
    pools = {}
    for label in LABELS:
        pool = sorted(
            (row for row in rows if row["label"] == label),
            key=lambda row: (row.get("patient_id", ""), row.get("slide_id", "")),
        )
        order = rng.permutation(len(pool)) if pool else []
        pools[label] = [pool[int(index)] for index in order]

    counts = {label: 0 for label in LABELS}
    positions = {label: 0 for label in LABELS}
    while any(counts[label] < PER_CLASS for label in LABELS):
        made_progress = False
        for label in LABELS:
            if counts[label] >= PER_CLASS:
                continue
            pool = pools[label]
            while positions[label] < len(pool):
                row = pool[positions[label]]
                positions[label] += 1
                patient_id = str(row.get("patient_id") or "").strip()
                if not patient_id or patient_id.upper() in NA_VALUES:
                    continue
                if patient_id in used_patients:
                    continue
                selected.append(row)
                counts[label] += 1
                used_patients.add(patient_id)
                made_progress = True
                break
        if not made_progress:
            break

    if counts != {"Tumor": PER_CLASS, "Normal": PER_CLASS}:
        available = {label: sum(1 for row in rows if row["label"] == label) for label in LABELS}
        raise RuntimeError(
            "Could not select 7 Tumor + 7 Normal slides with distinct patients. "
            f"Present eligible slides: {available}; selected: {counts}."
        )

    return sorted(selected, key=lambda row: (row["label"], row["slide_id"]))


def _selection_from_csv(selection_csv, inventory_rows, dataset_root, by_name, by_slide_prefix):
    with (
        Path(selection_csv)
        .expanduser()
        .resolve()
        .open("r", newline="", encoding="utf-8-sig") as handle
    ):
        requested = list(csv.DictReader(handle))
    if not requested or "slide_id" not in requested[0]:
        raise ValueError("selection CSV must contain a slide_id column")

    inventory_by_slide = {row["slide_id"]: row for row in inventory_rows}
    selected = []
    for request in requested:
        slide_id = str(request.get("slide_id") or "").strip()
        if slide_id not in inventory_by_slide:
            raise ValueError(f"slide_id not found in inventory: {slide_id}")
        row = dict(inventory_by_slide[slide_id])
        path, method = _resolve_inventory_row(
            row,
            dataset_root,
            by_name,
            by_slide_prefix,
        )
        if path is None:
            raise FileNotFoundError(f"Selected WSI is not present locally: {slide_id}")
        row["resolved_path"] = str(path)
        row["path_resolution_method"] = method
        selected.append(row)

    _validate_selection(selected)
    return sorted(selected, key=lambda row: (row["label"], row["slide_id"]))


def _validate_selection(selected):
    if len(selected) != 14:
        raise ValueError(f"Expected exactly 14 selected WSIs, got {len(selected)}.")
    counts = {label: 0 for label in LABELS}
    slide_ids = set()
    patient_ids = set()
    resolved_paths = set()
    for row in selected:
        label = row.get("label")
        if label not in counts:
            raise ValueError(f"Unexpected label for {row.get('slide_id')}: {label}")
        counts[label] += 1
        slide_id = str(row.get("slide_id") or "").strip()
        if not slide_id:
            raise ValueError("Selected rows must have a non-empty slide_id.")
        if slide_id in slide_ids:
            raise ValueError(f"Duplicate selected slide_id: {slide_id}")
        slide_ids.add(slide_id)
        patient_id = str(row.get("patient_id") or "").strip()
        if not patient_id or patient_id.upper() in NA_VALUES:
            raise ValueError(f"Missing patient_id for selected slide: {slide_id}")
        if patient_id in patient_ids:
            raise ValueError(
                f"Patient leakage in pilot selection: {patient_id} appears more than once."
            )
        patient_ids.add(patient_id)
        path_text = str(row.get("resolved_path") or "").strip()
        if not path_text:
            raise ValueError(f"Missing resolved_path for selected slide: {slide_id}")
        resolved = str(Path(path_text).resolve())
        if resolved in resolved_paths:
            raise ValueError(f"Duplicate physical WSI in selection: {resolved}")
        resolved_paths.add(resolved)
        if _parse_bool(row.get("duplicate")) is True:
            raise ValueError(f"Inventory marks selected slide as duplicate: {slide_id}")
        if _parse_bool(row.get("label_valid")) is False:
            raise ValueError(f"Inventory label is invalid for selected slide: {slide_id}")
    if counts != {"Tumor": 7, "Normal": 7}:
        raise ValueError(f"Selection must be 7 Tumor + 7 Normal, got {counts}.")


def _save_selection(selected, output_root):
    path = Path(output_root) / "selection.csv"
    fieldnames = [
        "slide_id",
        "patient_id",
        "label",
        "resolved_path",
        "path_resolution_method",
        "relative_path",
        "file_id",
        "file_name",
        "expected_md5",
        "source_size_bytes",
        "source_mtime_ns",
        "computed_md5",
    ]
    _write_csv_atomic(path, fieldnames, selected)
    return path


def _prepare_source_identities(selected, verify_md5):
    prepared = []
    for original in selected:
        row = dict(original)
        path = Path(row["resolved_path"])
        stat = path.stat()
        expected_size_text = str(row.get("expected_file_size_bytes") or "").strip()
        if expected_size_text and expected_size_text.upper() not in NA_VALUES:
            expected_size = int(expected_size_text)
            if stat.st_size != expected_size:
                raise ValueError(
                    f"Source size mismatch for {row['slide_id']}: "
                    f"expected {expected_size}, observed {stat.st_size}."
                )
        expected_md5 = str(row.get("expected_md5") or "").strip().lower()
        if expected_md5 and expected_md5.upper() not in NA_VALUES:
            if len(expected_md5) != 32 or any(
                character not in "0123456789abcdef" for character in expected_md5
            ):
                raise ValueError(f"Invalid expected MD5 for {row['slide_id']}: {expected_md5}")
        else:
            expected_md5 = ""
        computed_md5 = _hash_file(path, "md5") if verify_md5 else ""
        if verify_md5 and expected_md5 and computed_md5 != expected_md5:
            raise ValueError(
                f"Source MD5 mismatch for {row['slide_id']}: "
                f"expected {expected_md5}, observed {computed_md5}."
            )
        row.update(
            source_size_bytes=int(stat.st_size),
            source_mtime_ns=int(stat.st_mtime_ns),
            expected_md5=expected_md5,
            computed_md5=computed_md5,
        )
        prepared.append(row)
    return prepared


def _slide_seed(master_seed, slide_id):
    digest = hashlib.sha256(f"{master_seed}:{slide_id}".encode()).digest()
    # Keep the derived seed inside signed 63-bit range for portable HDF5 attrs.
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)


def _run_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_slide_dir_name(slide_id):
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in slide_id
    )


def _select_qc_indices(tissue_percentage, sample_count):
    total = len(tissue_percentage)
    if total == 0 or sample_count <= 0:
        return []
    count = min(int(sample_count), total)
    ranked = np.argsort(np.asarray(tissue_percentage, dtype=np.float64), kind="stable")
    positions = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    return sorted({int(ranked[position]) for position in positions})


def _quality_rows(images_uint8, coordinates, tissue_percentage, start):
    rows = []
    for local_index in range(images_uint8.shape[1]):
        patch_index = start + local_index
        original = images_uint8[0, local_index]
        for representation_index, representation in enumerate(REPRESENTATIONS):
            image = images_uint8[representation_index, local_index]
            image_float = image.astype(np.float32)
            gray = image_float.mean(axis=2)
            horizontal = np.diff(gray, axis=1)
            vertical = np.diff(gray, axis=0)
            maximum = image_float.max(axis=2)
            minimum = image_float.min(axis=2)
            saturation = np.divide(
                maximum - minimum,
                maximum,
                out=np.zeros_like(maximum),
                where=maximum > 0,
            )
            rows.append(
                {
                    "patch_index": patch_index,
                    "x": int(coordinates[patch_index, 0]),
                    "y": int(coordinates[patch_index, 1]),
                    "tissue_percentage": f"{float(tissue_percentage[patch_index]):.6f}",
                    "representation": representation,
                    "mean_intensity": f"{float(image_float.mean()):.6f}",
                    "std_intensity": f"{float(image_float.std()):.6f}",
                    "white_fraction": f"{float(np.all(image >= 245, axis=2).mean()):.8f}",
                    "dark_fraction": f"{float(np.all(image <= 10, axis=2).mean()):.8f}",
                    "clipped_fraction": f"{float(np.any((image == 0) | (image == 255), axis=2).mean()):.8f}",
                    "mean_saturation": f"{float(saturation.mean()):.8f}",
                    "gradient_energy": f"{float((np.square(horizontal).mean() + np.square(vertical).mean()) / 2.0):.6f}",
                    "identical_to_original": bool(np.array_equal(image, original)),
                }
            )
    return rows


def _save_augmentation_qc(images_uint8, output_path, patch_index=0):
    if images_uint8.shape[1] == 0:
        return None
    panels = [Image.fromarray(images_uint8[index, patch_index]) for index in range(3)]
    canvas = Image.new("RGB", (PATCH_SIZE * 3, PATCH_SIZE + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, panel) in enumerate(zip(REPRESENTATIONS, panels)):
        x = index * PATCH_SIZE
        canvas.paste(panel, (x, 40))
        draw.text((x + 8, 12), name, fill="black")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path.name)


def _save_augmentation_montage(qc_samples, output_path):
    if not qc_samples:
        return None
    row_height = PATCH_SIZE + 32
    canvas = Image.new(
        "RGB",
        (PATCH_SIZE * len(REPRESENTATIONS), row_height * len(qc_samples) + 36),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(REPRESENTATIONS):
        draw.text((column * PATCH_SIZE + 8, 10), name, fill="black")
    for row_index, (patch_index, tissue, images) in enumerate(qc_samples):
        top = 36 + row_index * row_height
        draw.text((8, top + 7), f"patch {patch_index} | tissue {tissue:.1f}%", fill="black")
        for column in range(len(REPRESENTATIONS)):
            canvas.paste(Image.fromarray(images[column]), (column * PATCH_SIZE, top + 32))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path.name)


def _quality_summary(rows):
    summary = {"row_count": len(rows), "representations": {}}
    numeric_fields = (
        "mean_intensity",
        "std_intensity",
        "white_fraction",
        "dark_fraction",
        "clipped_fraction",
        "mean_saturation",
        "gradient_energy",
    )
    for representation in REPRESENTATIONS:
        subset = [row for row in rows if row["representation"] == representation]
        entry = {
            "count": len(subset),
            "identical_to_original_count": sum(
                bool(row["identical_to_original"]) for row in subset
            ),
        }
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in subset], dtype=np.float64)
            entry[field] = {
                "min": float(values.min()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
        summary["representations"][representation] = entry
    return summary


def _validate_quality_artifacts(
    augmentation_dir,
    hdf5_path,
    *,
    patch_count,
    save_qc,
    qc_sample_count,
):
    """Recompute every saved quality row and validate QC image integrity."""
    augmentation_dir = Path(augmentation_dir)
    quality_path = augmentation_dir / "augmentation_quality.csv"
    summary_path = augmentation_dir / "augmentation_quality_summary.json"
    with quality_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        observed_rows = list(reader)
        observed_fields = reader.fieldnames
    expected_fields = [
        "patch_index",
        "x",
        "y",
        "tissue_percentage",
        "representation",
        "mean_intensity",
        "std_intensity",
        "white_fraction",
        "dark_fraction",
        "clipped_fraction",
        "mean_saturation",
        "gradient_energy",
        "identical_to_original",
    ]
    if observed_fields != expected_fields:
        raise ValueError("augmentation_quality.csv has an invalid header.")

    expected_rows = []
    with h5py.File(hdf5_path, "r") as file:
        coordinates = file["coords"][:]
        tissue = file["tissue_percentage"][:]
        for start in range(0, patch_count, 16):
            stop = min(start + 16, patch_count)
            expected_rows.extend(
                _quality_rows(file["images"][:, start:stop], coordinates, tissue, start)
            )
    if len(observed_rows) != len(expected_rows):
        raise ValueError("augmentation_quality.csv row count is invalid.")
    for index, (observed, expected) in enumerate(zip(observed_rows, expected_rows)):
        normalized = {key: str(value) for key, value in expected.items()}
        if observed != normalized:
            raise ValueError(f"augmentation_quality.csv row {index} does not match HDF5 content.")

    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if observed_summary != _quality_summary(expected_rows):
        raise ValueError("augmentation_quality_summary.json does not match quality rows.")

    if save_qc and patch_count:
        image_paths = [augmentation_dir / "augmentation_qc_patch_0000.png"]
        if qc_sample_count > 0:
            image_paths.append(augmentation_dir / "augmentation_qc_montage.png")
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image.verify()


def _create_hdf5(path, patch_count, metadata):
    file = h5py.File(path, "w")
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value)
        file.attrs[key] = value

    if patch_count == 0:
        file.create_dataset(
            "images",
            shape=(3, 0, PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
        )
    else:
        file.create_dataset(
            "images",
            shape=(3, patch_count, PATCH_SIZE, PATCH_SIZE, 3),
            dtype=np.uint8,
            chunks=(1, 1, PATCH_SIZE, PATCH_SIZE, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    file.create_dataset("coords", shape=(patch_count, 2), dtype=np.int64)
    file.create_dataset("tissue_percentage", shape=(patch_count,), dtype=np.float32)
    return file


def _save_augmented_views(
    *,
    slide_path,
    slide_id,
    patient_id,
    label,
    result,
    output_path,
    master_seed,
    chunk_size,
    save_qc,
    qc_sample_count,
    source_identity,
    run_fingerprint,
):
    valid = result["valid_patches"]
    patch_count = len(valid)
    coordinates = np.asarray([[item["x"], item["y"]] for item in valid], dtype=np.int64)
    if coordinates.size == 0:
        coordinates = np.empty((0, 2), dtype=np.int64)
    tissue_percentage = np.asarray([item["tissue_percentage"] for item in valid], dtype=np.float32)

    slide_seed = _slide_seed(master_seed, slide_id)
    rng = np.random.default_rng(slide_seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    if temporary_path.exists():
        temporary_path.unlink()

    metadata = {
        "format_version": HDF5_FORMAT_VERSION,
        "run_fingerprint": run_fingerprint,
        "slide_id": slide_id,
        "patient_id": patient_id,
        "label": label,
        "source_path": str(slide_path),
        "representations": list(REPRESENTATIONS),
        "image_layout": "representation,patch,height,width,channel",
        "stored_dtype": "uint8",
        "restored_tensor_layout": "representation,patch,channel,height,width",
        "restored_tensor_dtype": "float32",
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "patch_size": PATCH_SIZE,
        "target_magnification": result["resolution"]["target_magnification"],
        "target_downsample": result["resolution"]["target_downsample"],
        "master_seed": int(master_seed),
        "slide_seed": int(slide_seed),
        "patch_count": int(patch_count),
        "source_size_bytes": int(source_identity["source_size_bytes"]),
        "source_mtime_ns": int(source_identity["source_mtime_ns"]),
        "source_expected_md5": source_identity.get("expected_md5", ""),
        "source_computed_md5": source_identity.get("computed_md5", ""),
    }

    first_qc = None
    qc_indices = set(_select_qc_indices(tissue_percentage, qc_sample_count))
    qc_samples = []
    quality_rows = []
    try:
        with _create_hdf5(temporary_path, patch_count, metadata) as file:
            if patch_count:
                file["coords"][:] = coordinates
                file["tissue_percentage"][:] = tissue_percentage

                slide = open_slide(slide_path)
                try:
                    target_downsample = float(result["resolution"]["target_downsample"])
                    for start in range(0, patch_count, chunk_size):
                        stop = min(start + chunk_size, patch_count)
                        patches = [
                            extract_patch(
                                slide=slide,
                                level0_x=int(valid[index]["x"]),
                                level0_y=int(valid[index]["y"]),
                                target_downsample=target_downsample,
                                patch_size=PATCH_SIZE,
                            )
                            for index in range(start, stop)
                        ]
                        transformed = transform_batch(patches, rng=rng)
                        images_uint8 = normalized_views_to_uint8(transformed)

                        # This verifies that compact storage reproduces the exact
                        # tensor contract consumed by the KimiaNet branch.
                        restored = uint8_views_to_normalized(images_uint8)
                        if not torch.equal(transformed, restored):
                            raise RuntimeError("Internal lossless-storage round-trip check failed.")

                        file["images"][:, start:stop] = images_uint8
                        quality_rows.extend(
                            _quality_rows(
                                images_uint8,
                                coordinates,
                                tissue_percentage,
                                start,
                            )
                        )
                        if save_qc and first_qc is None:
                            qc_path = output_path.parent / "augmentation_qc_patch_0000.png"
                            first_qc = _save_augmentation_qc(
                                images_uint8,
                                qc_path,
                                patch_index=0,
                            )
                        if save_qc:
                            for local_index in range(stop - start):
                                patch_index = start + local_index
                                if patch_index in qc_indices:
                                    qc_samples.append(
                                        (
                                            patch_index,
                                            float(tissue_percentage[patch_index]),
                                            images_uint8[:, local_index].copy(),
                                        )
                                    )
                        print(
                            f"    augmentation: {stop}/{patch_count} patches",
                            flush=True,
                        )
                        del transformed, restored, images_uint8, patches
                finally:
                    slide.close()

        os.replace(temporary_path, output_path)
        quality_path = output_path.parent / "augmentation_quality.csv"
        quality_fields = [
            "patch_index",
            "x",
            "y",
            "tissue_percentage",
            "representation",
            "mean_intensity",
            "std_intensity",
            "white_fraction",
            "dark_fraction",
            "clipped_fraction",
            "mean_saturation",
            "gradient_energy",
            "identical_to_original",
        ]
        _write_csv_atomic(quality_path, quality_fields, quality_rows)
        quality_summary_path = output_path.parent / "augmentation_quality_summary.json"
        _write_json_atomic(_quality_summary(quality_rows), quality_summary_path)
        montage = None
        if save_qc:
            montage = _save_augmentation_montage(
                sorted(qc_samples, key=lambda item: item[0]),
                output_path.parent / "augmentation_qc_montage.png",
            )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "hdf5": str(output_path),
        "qc": first_qc,
        "qc_montage": montage,
        "quality_csv": str(quality_path.name),
        "quality_summary": str(quality_summary_path.name),
        "patch_count": patch_count,
        "slide_seed": slide_seed,
        "size_bytes": output_path.stat().st_size,
        "sha256": _hash_file(output_path),
    }


def _manifest_fieldnames():
    return [
        "slide_id",
        "patient_id",
        "label",
        "status",
        "slide_path",
        "candidate_tile_count",
        "valid_patch_count",
        "feature_extraction_ready",
        "augmentation_hdf5",
        "augmentation_size_bytes",
        "augmentation_sha256",
        "source_size_bytes",
        "source_md5",
        "augmentation_qc_montage",
        "augmentation_quality_csv",
        "preprocessing_seconds",
        "augmentation_seconds",
        "total_seconds",
        "error_type",
        "error_message",
    ]


def _write_manifest(rows, output_root):
    path = Path(output_root) / "manifest.csv"
    _write_csv_atomic(path, _manifest_fieldnames(), rows)
    return path


def _completed_payload(
    slide_dir,
    fingerprint,
    save_qc,
    *,
    preprocessing_config=None,
    source_identity=None,
    qc_sample_count=1,
):
    path = slide_dir / "complete.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("run_fingerprint") != fingerprint:
        return None
    hdf5_path = slide_dir / "augmentation" / "patch_views.h5"
    preprocessing_dir = slide_dir / "preprocessing"
    manifest_row = payload.get("manifest_row")
    if not isinstance(manifest_row, dict) or manifest_row.get("status") != "success":
        return None
    try:
        patch_count = int(manifest_row["valid_patch_count"])
        if patch_count < 0:
            return None
        if source_identity is not None and payload.get("source_identity") != source_identity:
            return None
        if payload.get("status") != "success":
            return None
        if not manifest_row.get("slide_id") or not manifest_row.get("patient_id"):
            return None
        if manifest_row.get("label") not in LABELS:
            return None
        if not hdf5_path.is_file():
            return None
        if hdf5_path.stat().st_size != int(manifest_row["augmentation_size_bytes"]):
            return None
        expected_sha256 = str(manifest_row.get("augmentation_sha256") or "")
        if len(expected_sha256) != 64 or _hash_file(hdf5_path) != expected_sha256:
            return None
        preprocessing_payload = validate_saved_outputs(
            preprocessing_dir,
            config=preprocessing_config,
            slide_path=manifest_row["slide_path"],
            expected_patch_count=patch_count,
        )
        if int(preprocessing_payload["result"]["candidate_tile_count"]) != int(
            manifest_row["candidate_tile_count"]
        ):
            return None
        coordinates = np.load(preprocessing_dir / "coordinates.npy", allow_pickle=False)
        validate_augmented_hdf5(
            hdf5_path,
            expected={
                "slide_id": manifest_row["slide_id"],
                "patient_id": manifest_row["patient_id"],
                "label": manifest_row["label"],
                "patch_count": patch_count,
                "run_fingerprint": fingerprint,
            },
        )
        with h5py.File(hdf5_path, "r") as file:
            if not np.array_equal(file["coords"][:], coordinates):
                return None
        _validate_quality_artifacts(
            slide_dir / "augmentation",
            hdf5_path,
            patch_count=patch_count,
            save_qc=save_qc,
            qc_sample_count=qc_sample_count,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, csv.Error):
        return None
    return payload


def _install_slide_directory(working_dir, final_dir):
    working_dir = Path(working_dir)
    final_dir = Path(final_dir)
    backup_dir = final_dir.parent / f".{final_dir.name}.backup.{os.getpid()}"
    if backup_dir.exists():
        raise RuntimeError(f"Refusing to overwrite stale backup directory: {backup_dir}")
    had_previous = final_dir.exists()
    if had_previous:
        os.replace(final_dir, backup_dir)
    try:
        os.replace(working_dir, final_dir)
    except Exception:
        if had_previous and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _validate_run_directories(dataset_root, output_root, project_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    project_root = Path(project_root).resolve()
    if output_root.is_relative_to(project_root) or project_root.is_relative_to(output_root):
        raise ValueError("Pilot outputs must be outside the repository in a separate directory.")
    if output_root.is_relative_to(dataset_root) or dataset_root.is_relative_to(output_root):
        raise ValueError("Pilot input and output directories must not overlap.")


def run(args):
    output_root = Path(args.output_dir).expanduser().resolve()
    _validate_run_directories(
        args.dataset_root,
        output_root,
        PROJECT_ROOT,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_run_lock(output_root)
    try:
        return _run_locked(args, output_root)
    finally:
        lock_path.unlink(missing_ok=True)


def _run_locked(args, output_root):
    dataset_root, local_paths, by_name, by_slide_prefix = _index_local_slides(args.dataset_root)
    inventory_rows = _load_inventory(args.inventory)

    if args.selection_csv:
        selected = _selection_from_csv(
            args.selection_csv,
            inventory_rows,
            dataset_root,
            by_name,
            by_slide_prefix,
        )
    else:
        present = _eligible_present_rows(
            inventory_rows,
            dataset_root,
            by_name,
            by_slide_prefix,
        )
        selected = _select_balanced(present, args.seed)
        _validate_selection(selected)

    selected = _prepare_source_identities(selected, args.verify_source_md5)
    selection_path = _save_selection(selected, output_root)

    preprocessing_config = build_config(
        target_magnification=args.target_magnification,
        patch_size=PATCH_SIZE,
        min_tissue_percentage=args.min_tissue_percentage,
        color_distance_threshold=args.color_distance_threshold,
        thumbnail_max_size=args.thumbnail_max_size,
        save_qc=args.save_qc,
    )
    project_root = PROJECT_ROOT
    inventory_path = Path(args.inventory).expanduser().resolve()
    source_identities = [
        {
            "slide_id": row["slide_id"],
            "patient_id": row["patient_id"],
            "label": row["label"],
            "resolved_path": row["resolved_path"],
            "source_size_bytes": row["source_size_bytes"],
            "source_mtime_ns": row["source_mtime_ns"],
            "expected_md5": row.get("expected_md5", ""),
            "computed_md5": row.get("computed_md5", ""),
        }
        for row in selected
    ]
    run_config = {
        "pilot_version": PILOT_VERSION,
        "dataset_root": str(dataset_root),
        "inventory": str(inventory_path),
        "inventory_sha256": _hash_file(inventory_path),
        "selection_csv": str(Path(args.selection_csv).expanduser().resolve())
        if args.selection_csv
        else None,
        "seed": int(args.seed),
        "chunk_size": int(args.chunk_size),
        "save_qc": bool(args.save_qc),
        "qc_sample_count": int(args.qc_sample_count),
        "verify_source_md5": bool(args.verify_source_md5),
        "preprocessing": preprocessing_config,
        "selection_rule": "7 Tumor + 7 Normal, distinct patients"
        if not args.selection_csv
        else "explicit selection CSV",
        "selected_slide_ids": [row["slide_id"] for row in selected],
        "source_identities": source_identities,
        "pipeline_code_sha256": _code_fingerprint(project_root),
        "runtime": _runtime_versions(),
    }
    fingerprint = _run_fingerprint(run_config)
    run_config["run_fingerprint"] = fingerprint
    run_config["preprocessing_fingerprint"] = config_fingerprint(preprocessing_config)
    run_config["local_wsi_files_discovered"] = len(local_paths)
    _write_json_atomic(run_config, output_root / "run_config.json")

    print(f"Found {len(local_paths)} local WSI file(s) under {dataset_root}")
    print(f"Selected cohort: 7 Tumor + 7 Normal -> {selection_path}")
    print(f"Output: {output_root}")
    print(f"Run fingerprint: {fingerprint}")

    rows = []
    failures = 0
    for index, row in enumerate(selected, start=1):
        slide_id = row["slide_id"]
        patient_id = row.get("patient_id", "")
        label = row["label"]
        slide_path = Path(row["resolved_path"])
        slide_dir = output_root / "slides" / _safe_slide_dir_name(slide_id)

        print(f"\n[{index}/14] {label} | {slide_id}\n  {slide_path}")

        if not args.force:
            completed = _completed_payload(
                slide_dir,
                fingerprint,
                args.save_qc,
                preprocessing_config=preprocessing_config,
                source_identity=source_identities[index - 1],
                qc_sample_count=args.qc_sample_count,
            )
            if completed is not None:
                manifest_row = completed["manifest_row"]
                rows.append(manifest_row)
                _write_manifest(rows, output_root)
                print("  Resume: already complete with matching configuration.")
                continue

        working_dir = (
            output_root / "slides" / f".{_safe_slide_dir_name(slide_id)}.working.{os.getpid()}"
        )
        if working_dir.exists():
            raise RuntimeError(f"Refusing to overwrite stale working directory: {working_dir}")
        working_dir.mkdir(parents=True, exist_ok=False)
        started_total = time.perf_counter()
        manifest_row = {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "label": label,
            "status": "failed",
            "slide_path": str(slide_path),
            "candidate_tile_count": "",
            "valid_patch_count": "",
            "feature_extraction_ready": "",
            "augmentation_hdf5": "",
            "augmentation_size_bytes": "",
            "augmentation_sha256": "",
            "source_size_bytes": row["source_size_bytes"],
            "source_md5": row.get("computed_md5") or row.get("expected_md5", ""),
            "augmentation_qc_montage": "",
            "augmentation_quality_csv": "",
            "preprocessing_seconds": "",
            "augmentation_seconds": "",
            "total_seconds": "",
            "error_type": "",
            "error_message": "",
        }

        try:
            started = time.perf_counter()
            result = preprocess_slide(
                slide_path=slide_path,
                target_magnification=args.target_magnification,
                patch_size=PATCH_SIZE,
                min_tissue_percentage=args.min_tissue_percentage,
                color_distance_threshold=args.color_distance_threshold,
                thumbnail_max_size=args.thumbnail_max_size,
            )
            preprocessing_seconds = time.perf_counter() - started
            preprocessing_dir = working_dir / "preprocessing"
            save_slide_outputs(
                result=result,
                slide_path=slide_path,
                output_dir=preprocessing_dir,
                config=preprocessing_config,
                elapsed_seconds=preprocessing_seconds,
                save_qc=args.save_qc,
            )

            print(
                f"  preprocessing: {result['candidate_tile_count']} candidates -> "
                f"{result['valid_patch_count']} valid patches "
                f"({preprocessing_seconds:.1f}s)"
            )

            started = time.perf_counter()
            augmentation_path = working_dir / "augmentation" / "patch_views.h5"
            augmentation = _save_augmented_views(
                slide_path=slide_path,
                slide_id=slide_id,
                patient_id=patient_id,
                label=label,
                result=result,
                output_path=augmentation_path,
                master_seed=args.seed,
                chunk_size=args.chunk_size,
                save_qc=args.save_qc,
                qc_sample_count=args.qc_sample_count,
                source_identity=row,
                run_fingerprint=fingerprint,
            )
            augmentation_seconds = time.perf_counter() - started
            total_seconds = time.perf_counter() - started_total

            manifest_row.update(
                {
                    "status": "success",
                    "candidate_tile_count": result["candidate_tile_count"],
                    "valid_patch_count": result["valid_patch_count"],
                    "feature_extraction_ready": result["feature_extraction_ready"],
                    "augmentation_hdf5": str(slide_dir / "augmentation" / "patch_views.h5"),
                    "augmentation_size_bytes": augmentation["size_bytes"],
                    "augmentation_sha256": augmentation["sha256"],
                    "augmentation_qc_montage": augmentation.get("qc_montage") or "",
                    "augmentation_quality_csv": augmentation["quality_csv"],
                    "preprocessing_seconds": f"{preprocessing_seconds:.6f}",
                    "augmentation_seconds": f"{augmentation_seconds:.6f}",
                    "total_seconds": f"{total_seconds:.6f}",
                }
            )
            complete = {
                "status": "success",
                "run_fingerprint": fingerprint,
                "slide_seed": augmentation["slide_seed"],
                "source_identity": source_identities[index - 1],
                "manifest_row": manifest_row,
            }
            _write_json_atomic(complete, working_dir / "complete.json")
            if (
                _completed_payload(
                    working_dir,
                    fingerprint,
                    args.save_qc,
                    preprocessing_config=preprocessing_config,
                    source_identity=source_identities[index - 1],
                    qc_sample_count=args.qc_sample_count,
                )
                is None
            ):
                raise RuntimeError("New slide artifacts failed post-write integrity validation.")
            _install_slide_directory(working_dir, slide_dir)
            failure_path = output_root / "failures" / f"{_safe_slide_dir_name(slide_id)}.json"
            failure_path.unlink(missing_ok=True)
            print(
                f"  saved: {slide_dir / 'augmentation' / 'patch_views.h5'} "
                f"({augmentation['size_bytes'] / (1024**2):.1f} MiB)"
            )

        except Exception as error:
            failures += 1
            total_seconds = time.perf_counter() - started_total
            manifest_row.update(
                {
                    "status": "failed",
                    "total_seconds": f"{total_seconds:.6f}",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            _write_json_atomic(
                {
                    "status": "failed",
                    "run_fingerprint": fingerprint,
                    "slide_id": slide_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
                output_root / "failures" / f"{_safe_slide_dir_name(slide_id)}.json",
            )
            if working_dir.exists():
                shutil.rmtree(working_dir)
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)

        rows.append(manifest_row)
        _write_manifest(rows, output_root)

    manifest_path = _write_manifest(rows, output_root)
    successes = sum(row["status"] == "success" for row in rows)
    print("\n=== PILOT SUMMARY ===")
    print(f"Successful: {successes}/14")
    print(f"Failed: {failures}")
    print(f"Manifest: {manifest_path}")
    print(f"Selection: {selection_path}")

    if failures:
        return 2
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing preprocessing + three-view augmentation pipeline "
            "on a balanced 14-WSI pilot cohort (7 Tumor, 7 Normal)."
        )
    )
    parser.add_argument("dataset_root", help="Root directory containing downloaded WSIs.")
    parser.add_argument("output_dir", help="Output directory for the 14-WSI pilot run.")
    parser.add_argument(
        "--inventory",
        default="metadata/dataset_inventory.csv",
        help="Dataset inventory CSV with slide_id/patient_id/label/relative_path.",
    )
    parser.add_argument(
        "--selection-csv",
        help=(
            "Optional explicit 14-slide CSV containing slide_id. When omitted, "
            "the script deterministically selects 7 Tumor + 7 Normal local slides."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Selection and augmentation master seed."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Patches transformed at once; 8 is ~72 MiB for output tensors.",
    )
    parser.add_argument(
        "--qc-sample-count",
        type=int,
        default=6,
        help="Number of tissue-stratified patches per slide in the augmentation QC montage.",
    )
    parser.add_argument("--target-magnification", type=float, default=5.0)
    parser.add_argument("--min-tissue-percentage", type=float, default=20.0)
    parser.add_argument("--color-distance-threshold", type=float, default=35.0)
    parser.add_argument("--thumbnail-max-size", type=int, default=1024)
    parser.add_argument(
        "--no-qc",
        dest="save_qc",
        action="store_false",
        help="Do not save preprocessing QC or one augmentation preview per slide.",
    )
    parser.set_defaults(save_qc=True)
    parser.add_argument(
        "--verify-source-md5",
        action="store_true",
        help="Hash each selected WSI and require it to match the inventory MD5 before processing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run slides even if a matching complete result already exists.",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")
    if args.qc_sample_count < 0:
        parser.error("--qc-sample-count must be >= 0")
    if args.save_qc and args.qc_sample_count == 0:
        parser.error("--qc-sample-count must be > 0 unless --no-qc is used")
    if not math.isfinite(args.target_magnification) or args.target_magnification <= 0:
        parser.error("--target-magnification must be > 0")
    if not math.isfinite(args.min_tissue_percentage) or not 0 <= args.min_tissue_percentage <= 100:
        parser.error("--min-tissue-percentage must be finite and between 0 and 100")
    if not math.isfinite(args.color_distance_threshold) or args.color_distance_threshold < 0:
        parser.error("--color-distance-threshold must be finite and >= 0")
    if args.thumbnail_max_size <= 0:
        parser.error("--thumbnail-max-size must be > 0")
    return args


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
