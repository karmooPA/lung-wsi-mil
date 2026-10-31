#!/usr/bin/env python3
"""Independently verify a completed 14-WSI preprocessing/augmentation pilot."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_14_wsi_pilot import _code_fingerprint
from src.augmentation.data_augmentations import PATCH_SIZE
from src.augmentation.storage import REPRESENTATIONS, load_augmented_chunk
from lung_wsi_mil.preprocessing.batch_preprocessing import validate_saved_outputs
from lung_wsi_mil.preprocessing.patch_extraction import extract_patch
from lung_wsi_mil.preprocessing.wsi_reader import open_slide


QUALITY_NUMERIC_FIELDS = (
    "mean_intensity",
    "std_intensity",
    "white_fraction",
    "dark_fraction",
    "clipped_fraction",
    "mean_saturation",
    "gradient_energy",
)
SOURCE_IDENTITY_FIELDS = (
    "slide_id",
    "patient_id",
    "label",
    "resolved_path",
    "source_size_bytes",
    "source_mtime_ns",
    "expected_md5",
    "computed_md5",
)
POST_FINGERPRINT_FIELDS = {
    "run_fingerprint",
    "preprocessing_fingerprint",
    "local_wsi_files_discovered",
}


def _hash_file(path, algorithm="sha256", block_size=4 * 1024 * 1024):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_text_atomic(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sample_indices(count, requested):
    if count <= 0 or requested <= 0:
        return []
    return sorted(set(np.rint(np.linspace(0, count - 1, min(count, requested))).astype(int)))


def _finite_float(value, field):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value}")
    return result


def _boolean(value, field):
    if isinstance(value, bool):
        return value
    normalized = _text(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean {field}: {value}")


def _check_current_pipeline_code(run_config, project_root=PROJECT_ROOT):
    expected = _text(run_config.get("pipeline_code_sha256")).strip().lower()
    observed = _code_fingerprint(project_root)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        return observed, "run_config pipeline_code_sha256 is missing or invalid"
    if observed != expected:
        return (
            observed,
            "Current pipeline code SHA256 does not match the code that generated "
            f"the outputs (expected {expected}, observed {observed})",
        )
    return observed, None


def _safe_slide_dir_name(slide_id):
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in slide_id
    )


def _text(value):
    return "" if value is None else str(value)


def _resolved_text(value):
    return str(Path(str(value)).expanduser().resolve())


def _recompute_run_fingerprint(run_config):
    payload = {
        key: value
        for key, value in run_config.items()
        if key not in POST_FINGERPRINT_FIELDS
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _source_identity_map(run_config):
    identities = run_config.get("source_identities")
    if not isinstance(identities, list):
        raise ValueError("run_config source_identities must be a list")
    mapping = {}
    for identity in identities:
        if not isinstance(identity, dict):
            raise ValueError("run_config source identity must be an object")
        slide_id = _text(identity.get("slide_id")).strip()
        if not slide_id:
            raise ValueError("run_config source identity has an empty slide_id")
        if slide_id in mapping:
            raise ValueError(f"Duplicate run_config source identity: {slide_id}")
        missing = [field for field in SOURCE_IDENTITY_FIELDS if field not in identity]
        if missing:
            raise ValueError(
                f"run_config source identity for {slide_id} is missing: "
                + ", ".join(missing)
            )
        mapping[slide_id] = identity
    return mapping


def _compare_source_identities(observed, expected, context):
    for field in SOURCE_IDENTITY_FIELDS:
        if field in {"source_size_bytes", "source_mtime_ns"}:
            try:
                matches = int(observed.get(field, -1)) == int(expected[field])
            except (TypeError, ValueError):
                matches = False
        elif field == "resolved_path":
            try:
                matches = _resolved_text(observed.get(field, "")) == _resolved_text(
                    expected[field]
                )
            except (OSError, ValueError):
                matches = False
        elif field in {"expected_md5", "computed_md5"}:
            matches = _text(observed.get(field)).lower() == _text(expected[field]).lower()
        else:
            matches = _text(observed.get(field)) == _text(expected[field])
        if not matches:
            raise ValueError(f"{context} mismatch for {field}")


def _validate_control_files(selection, manifest, run_config):
    """Cross-check the three control surfaces before inspecting slide payloads."""
    errors = []
    selection_ids = [_text(row.get("slide_id")).strip() for row in selection]
    manifest_ids = [_text(row.get("slide_id")).strip() for row in manifest]
    if not all(selection_ids) or len(set(selection_ids)) != len(selection_ids):
        errors.append("Selection slide IDs are empty or duplicated")
    if not all(manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        errors.append("Manifest slide IDs are empty or duplicated")
    if selection_ids != manifest_ids:
        errors.append("Selection and manifest slide IDs/order differ")

    configured_ids = run_config.get("selected_slide_ids")
    if configured_ids != selection_ids:
        errors.append("run_config selected_slide_ids does not match selection.csv")

    observed_fingerprint = _text(run_config.get("run_fingerprint"))
    if observed_fingerprint != _recompute_run_fingerprint(run_config):
        errors.append("run_config fingerprint does not match its contents")

    try:
        identities = _source_identity_map(run_config)
    except ValueError as error:
        errors.append(str(error))
        identities = {}
    if set(identities) != set(selection_ids):
        errors.append("run_config source identities do not match selected slide IDs")

    manifest_by_id = {
        _text(row.get("slide_id")).strip(): row
        for row in manifest
        if _text(row.get("slide_id")).strip()
    }
    for selected in selection:
        slide_id = _text(selected.get("slide_id")).strip()
        identity = identities.get(slide_id)
        row = manifest_by_id.get(slide_id)
        if identity is None or row is None:
            continue
        try:
            _compare_source_identities(selected, identity, f"selection {slide_id}")
        except ValueError as error:
            errors.append(str(error))
        for field in ("patient_id", "label"):
            if _text(row.get(field)) != _text(selected.get(field)):
                errors.append(f"Selection/manifest mismatch for {slide_id} {field}")
        try:
            if _resolved_text(row.get("slide_path", "")) != _resolved_text(
                selected.get("resolved_path", "")
            ):
                errors.append(f"Selection/manifest path mismatch for {slide_id}")
        except (OSError, ValueError):
            errors.append(f"Selection/manifest path is invalid for {slide_id}")
        try:
            if int(row.get("source_size_bytes", -1)) != int(
                selected.get("source_size_bytes", -2)
            ):
                errors.append(f"Selection/manifest source size mismatch for {slide_id}")
        except (TypeError, ValueError):
            errors.append(f"Selection/manifest source size is invalid for {slide_id}")
        expected_md5 = _text(selected.get("computed_md5")).lower() or _text(
            selected.get("expected_md5")
        ).lower()
        if _text(row.get("source_md5")).lower() != expected_md5:
            errors.append(f"Selection/manifest source MD5 mismatch for {slide_id}")
    return errors


def _computed_quality_rows(images_uint8, coordinates, tissue_percentage, start):
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
                    "patch_index": str(patch_index),
                    "x": str(int(coordinates[patch_index, 0])),
                    "y": str(int(coordinates[patch_index, 1])),
                    "tissue_percentage": f"{float(tissue_percentage[patch_index]):.6f}",
                    "representation": representation,
                    "mean_intensity": f"{float(image_float.mean()):.6f}",
                    "std_intensity": f"{float(image_float.std()):.6f}",
                    "white_fraction": f"{float(np.all(image >= 245, axis=2).mean()):.8f}",
                    "dark_fraction": f"{float(np.all(image <= 10, axis=2).mean()):.8f}",
                    "clipped_fraction": f"{float(np.any((image == 0) | (image == 255), axis=2).mean()):.8f}",
                    "mean_saturation": f"{float(saturation.mean()):.8f}",
                    "gradient_energy": f"{float((np.square(horizontal).mean() + np.square(vertical).mean()) / 2.0):.6f}",
                    "identical_to_original": str(bool(np.array_equal(image, original))),
                }
            )
    return rows


def _computed_quality_summary(rows):
    summary = {"row_count": len(rows), "representations": {}}
    for representation in REPRESENTATIONS:
        subset = [row for row in rows if row["representation"] == representation]
        entry = {
            "count": len(subset),
            "identical_to_original_count": sum(
                row["identical_to_original"] == "True" for row in subset
            ),
        }
        for field in QUALITY_NUMERIC_FIELDS:
            values = np.asarray([float(row[field]) for row in subset], dtype=np.float64)
            entry[field] = {
                "min": float(values.min()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "max": float(values.max()) if len(values) else None,
            }
        summary["representations"][representation] = entry
    return summary


def _validate_quality_rows(observed, expected, slide_id):
    if len(observed) != len(expected):
        raise ValueError(f"Quality metric row count mismatch for {slide_id}")
    exact_fields = (
        "patch_index",
        "x",
        "y",
        "representation",
        "identical_to_original",
    )
    numeric_fields = ("tissue_percentage",) + QUALITY_NUMERIC_FIELDS
    for row_index, (actual, wanted) in enumerate(zip(observed, expected)):
        for field in exact_fields:
            if _text(actual.get(field)) != wanted[field]:
                raise ValueError(
                    f"Quality row {row_index} {field} mismatch for {slide_id}"
                )
        for field in numeric_fields:
            actual_value = _finite_float(actual.get(field), field)
            wanted_value = float(wanted[field])
            precision = 8 if field in {
                "white_fraction",
                "dark_fraction",
                "clipped_fraction",
                "mean_saturation",
            } else 6
            if not math.isclose(
                actual_value,
                wanted_value,
                rel_tol=0,
                abs_tol=0.5 * (10 ** -precision),
            ):
                raise ValueError(
                    f"Quality row {row_index} {field} mismatch for {slide_id}"
                )


def _verify_slide(row, output_root, run_config, verify_source_md5, source_sample_count):
    slide_id = row["slide_id"]
    label = row["label"]
    patient_id = row["patient_id"]
    expected_count = int(row["valid_patch_count"])
    source_path = Path(row["slide_path"])
    slide_dir = output_root / "slides" / _safe_slide_dir_name(slide_id)
    preprocessing_dir = slide_dir / "preprocessing"
    augmentation_dir = slide_dir / "augmentation"
    hdf5_path = augmentation_dir / "patch_views.h5"

    recorded_hdf5_path = _text(row.get("augmentation_hdf5")).strip()
    if not recorded_hdf5_path:
        raise ValueError(f"Manifest augmentation_hdf5 path is missing for {slide_id}")
    if _resolved_text(recorded_hdf5_path) != _resolved_text(hdf5_path):
        raise ValueError(f"Manifest augmentation_hdf5 path mismatch for {slide_id}")

    complete = json.loads((slide_dir / "complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "success":
        raise ValueError(f"Completion marker is not successful for {slide_id}")
    if complete.get("run_fingerprint") != run_config["run_fingerprint"]:
        raise ValueError(f"Run fingerprint mismatch for {slide_id}")
    complete_manifest = complete.get("manifest_row")
    if not isinstance(complete_manifest, dict):
        raise ValueError(f"Completion marker has no manifest row for {slide_id}")
    for field, value in row.items():
        if _text(complete_manifest.get(field)) != _text(value):
            raise ValueError(f"Completion/manifest mismatch for {slide_id} {field}")

    identities = _source_identity_map(run_config)
    source_identity = identities.get(slide_id)
    if source_identity is None:
        raise ValueError(f"No run_config source identity for {slide_id}")
    complete_identity = complete.get("source_identity")
    if not isinstance(complete_identity, dict):
        raise ValueError(f"Completion marker has no source identity for {slide_id}")
    _compare_source_identities(
        complete_identity,
        source_identity,
        f"Completion source identity for {slide_id}",
    )

    preprocessing_config = run_config.get("preprocessing")
    if not isinstance(preprocessing_config, dict):
        raise ValueError("run_config preprocessing must be an object")
    result_payload = validate_saved_outputs(
        preprocessing_dir,
        config=preprocessing_config,
        slide_path=source_path,
        expected_patch_count=expected_count,
    )
    result = result_payload["result"]
    if int(result["candidate_tile_count"]) != int(row["candidate_tile_count"]):
        raise ValueError(
            f"Preprocessing/manifest candidate tile count mismatch for {slide_id}"
        )
    result_ready = _boolean(
        result.get("feature_extraction_ready"),
        "preprocessing feature_extraction_ready",
    )
    manifest_ready = _boolean(
        row.get("feature_extraction_ready"),
        "manifest feature_extraction_ready",
    )
    if result_ready != manifest_ready:
        raise ValueError(
            f"Preprocessing/manifest feature_extraction_ready mismatch for {slide_id}"
        )
    resolution = result.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError(f"Preprocessing resolution is missing for {slide_id}")
    preprocessing_target_magnification = _finite_float(
        resolution.get("target_magnification"),
        "preprocessing target_magnification",
    )
    preprocessing_target_downsample = _finite_float(
        resolution.get("target_downsample"),
        "preprocessing target_downsample",
    )
    if preprocessing_target_magnification <= 0 or preprocessing_target_downsample <= 0:
        raise ValueError(f"Preprocessing target resolution is invalid for {slide_id}")
    coordinates = np.load(preprocessing_dir / "coordinates.npy", allow_pickle=False)
    if coordinates.shape != (expected_count, 2) or coordinates.dtype != np.int64:
        raise ValueError(f"Invalid coordinates.npy contract for {slide_id}: {coordinates.shape}/{coordinates.dtype}")
    if len(np.unique(coordinates, axis=0)) != expected_count:
        raise ValueError(f"Duplicate patch coordinates for {slide_id}")

    patch_rows = _read_csv(preprocessing_dir / "valid_patches.csv")
    if len(patch_rows) != expected_count:
        raise ValueError(f"valid_patches.csv count mismatch for {slide_id}")
    csv_coordinates = np.asarray(
        [[int(item["x"]), int(item["y"])] for item in patch_rows],
        dtype=np.int64,
    )
    if not len(patch_rows):
        csv_coordinates = np.empty((0, 2), dtype=np.int64)
    np.testing.assert_array_equal(csv_coordinates, coordinates)
    tissue = np.asarray([_finite_float(item["tissue_percentage"], "tissue_percentage") for item in patch_rows])
    threshold = float(run_config["preprocessing"]["min_tissue_percentage"])
    if expected_count and not bool(np.all(tissue > threshold)):
        raise ValueError(f"Stored patch does not satisfy strict tissue threshold for {slide_id}")

    expected_sha256 = row["augmentation_sha256"]
    observed_sha256 = _hash_file(hdf5_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"HDF5 SHA256 mismatch for {slide_id}")
    if hdf5_path.stat().st_size != int(row["augmentation_size_bytes"]):
        raise ValueError(f"HDF5 size mismatch for {slide_id}")

    quality_rows = _read_csv(augmentation_dir / "augmentation_quality.csv")
    identical = {"Aug1": 0, "Aug2": 0}
    recomputed_quality_rows = []
    with h5py.File(hdf5_path, "r") as file:
        if set(file.keys()) != {"images", "coords", "tissue_percentage"}:
            raise ValueError(f"Unexpected HDF5 datasets for {slide_id}: {sorted(file.keys())}")
        if file["images"].shape != (3, expected_count, PATCH_SIZE, PATCH_SIZE, 3):
            raise ValueError(f"Invalid image shape for {slide_id}: {file['images'].shape}")
        if file["images"].dtype != np.dtype(np.uint8):
            raise ValueError(f"Invalid image dtype for {slide_id}: {file['images'].dtype}")
        if file["coords"].shape != (expected_count, 2) or file["coords"].dtype != np.dtype(np.int64):
            raise ValueError(f"Invalid coordinate dataset for {slide_id}")
        if file["tissue_percentage"].shape != (expected_count,) or file["tissue_percentage"].dtype != np.dtype(np.float32):
            raise ValueError(f"Invalid tissue dataset for {slide_id}")
        if str(file.attrs.get("slide_id", "")) != slide_id or str(file.attrs.get("label", "")) != label:
            raise ValueError(f"HDF5 identity mismatch for {slide_id}")
        if str(file.attrs.get("patient_id", "")) != patient_id:
            raise ValueError(f"HDF5 patient identity mismatch for {slide_id}")
        if str(file.attrs.get("run_fingerprint", "")) != run_config["run_fingerprint"]:
            raise ValueError(f"HDF5 fingerprint mismatch for {slide_id}")
        if _resolved_text(file.attrs.get("source_path", "")) != _resolved_text(
            source_identity["resolved_path"]
        ):
            raise ValueError(f"HDF5 source path mismatch for {slide_id}")
        hdf5_target_magnification = _finite_float(
            file.attrs.get("target_magnification"),
            "HDF5 target_magnification",
        )
        hdf5_target_downsample = _finite_float(
            file.attrs.get("target_downsample"),
            "HDF5 target_downsample",
        )
        if not math.isclose(
            hdf5_target_magnification,
            preprocessing_target_magnification,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"HDF5/preprocessing target magnification mismatch for {slide_id}"
            )
        if not math.isclose(
            hdf5_target_downsample,
            preprocessing_target_downsample,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"HDF5/preprocessing target downsample mismatch for {slide_id}"
            )
        if int(file.attrs.get("source_size_bytes", -1)) != int(
            source_identity["source_size_bytes"]
        ):
            raise ValueError(f"HDF5 source size mismatch for {slide_id}")
        if int(file.attrs.get("source_mtime_ns", -1)) != int(
            source_identity["source_mtime_ns"]
        ):
            raise ValueError(f"HDF5 source mtime mismatch for {slide_id}")
        if str(file.attrs.get("source_expected_md5", "")).lower() != _text(
            source_identity["expected_md5"]
        ).lower():
            raise ValueError(f"HDF5 expected source MD5 mismatch for {slide_id}")
        if str(file.attrs.get("source_computed_md5", "")).lower() != _text(
            source_identity["computed_md5"]
        ).lower():
            raise ValueError(f"HDF5 computed source MD5 mismatch for {slide_id}")
        np.testing.assert_array_equal(file["coords"][:], coordinates)
        np.testing.assert_allclose(file["tissue_percentage"][:], tissue.astype(np.float32), rtol=0, atol=1e-5)
        for start in range(0, expected_count, 16):
            stop = min(start + 16, expected_count)
            images = file["images"][:, start:stop]
            identical["Aug1"] += int(np.all(images[1] == images[0], axis=(1, 2, 3)).sum())
            identical["Aug2"] += int(np.all(images[2] == images[0], axis=(1, 2, 3)).sum())
            recomputed_quality_rows.extend(
                _computed_quality_rows(images, coordinates, tissue.astype(np.float32), start)
            )

        if _resolved_text(source_path) != _resolved_text(source_identity["resolved_path"]):
            raise ValueError(f"Source path mismatch for {slide_id}")
        source_stat = source_path.stat()
        if source_stat.st_size != int(row["source_size_bytes"]):
            raise ValueError(f"Source WSI size changed for {slide_id}")
        if source_stat.st_mtime_ns != int(source_identity["source_mtime_ns"]):
            raise ValueError(f"Source WSI mtime changed for {slide_id}")
        source_md5 = ""
        if verify_source_md5:
            source_md5 = _hash_file(source_path, "md5")
            if source_md5 != row["source_md5"]:
                raise ValueError(f"Source WSI MD5 mismatch for {slide_id}")

        sample_indices = _sample_indices(expected_count, source_sample_count)
        slide = open_slide(source_path)
        try:
            width, height = slide.dimensions
            if expected_count and (
                bool(np.any(coordinates < 0))
                or bool(np.any(coordinates[:, 0] >= width))
                or bool(np.any(coordinates[:, 1] >= height))
            ):
                raise ValueError(f"Out-of-bounds Level-0 coordinate for {slide_id}")
            for index in sample_indices:
                patch = extract_patch(
                    slide=slide,
                    level0_x=int(coordinates[index, 0]),
                    level0_y=int(coordinates[index, 1]),
                    target_downsample=hdf5_target_downsample,
                    patch_size=PATCH_SIZE,
                )
                stored_original = file["images"][0, index]
                if not np.array_equal(stored_original, np.asarray(patch, dtype=np.uint8)):
                    raise ValueError(f"Stored Original does not match source extraction for {slide_id} patch {index}")
        finally:
            slide.close()

    if expected_count:
        for representation in ("Aug1", "Aug2"):
            if identical[representation] == expected_count:
                raise ValueError(
                    f"All {representation} views equal Original for {slide_id}"
                )

    _validate_quality_rows(quality_rows, recomputed_quality_rows, slide_id)
    quality_summary = json.loads(
        (augmentation_dir / "augmentation_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_quality_summary = _computed_quality_summary(recomputed_quality_rows)
    if quality_summary != expected_quality_summary:
        raise ValueError(f"Quality summary mismatch for {slide_id}")

    qc_paths = []
    if bool(run_config.get("save_qc")):
        qc_paths.extend(
            (
                preprocessing_dir / "qc" / "thumbnail.png",
                preprocessing_dir / "qc" / "filtered_mask.png",
                preprocessing_dir / "qc" / "valid_patches_overlay.png",
            )
        )
        if expected_count:
            qc_paths.append(augmentation_dir / "augmentation_qc_patch_0000.png")
            if int(run_config.get("qc_sample_count", 0)) > 0:
                qc_paths.append(augmentation_dir / "augmentation_qc_montage.png")
    for path in qc_paths:
        with Image.open(path) as image:
            image.verify()

    if expected_count:
        restored, restored_coordinates, restored_tissue = load_augmented_chunk(
            hdf5_path, 0, min(expected_count, 2)
        )
        if restored.dtype != torch.float32 or not bool(torch.isfinite(restored).all()):
            raise ValueError(f"Restored tensors are invalid for {slide_id}")
        np.testing.assert_array_equal(restored_coordinates, coordinates[: len(restored_coordinates)])
        np.testing.assert_allclose(restored_tissue, tissue[: len(restored_tissue)], rtol=0, atol=1e-5)

    return {
        "slide_id": slide_id,
        "patient_id": patient_id,
        "label": label,
        "patch_count": expected_count,
        "candidate_tile_count": int(row["candidate_tile_count"]),
        "hdf5_bytes": hdf5_path.stat().st_size,
        "hdf5_sha256": observed_sha256,
        "source_md5": source_md5 or row["source_md5"],
        "source_reextract_samples": len(_sample_indices(expected_count, source_sample_count)),
        "augmented_identical_to_original": identical,
        "tissue_percentage": {
            "min": float(tissue.min()) if len(tissue) else None,
            "median": float(np.median(tissue)) if len(tissue) else None,
            "max": float(tissue.max()) if len(tissue) else None,
        },
    }


def verify(output_dir, verify_source_md5=False, source_sample_count=3):
    output_root = Path(output_dir).expanduser().resolve()
    errors = []
    details = []
    run_config = json.loads((output_root / "run_config.json").read_text(encoding="utf-8"))
    selection = _read_csv(output_root / "selection.csv")
    manifest = _read_csv(output_root / "manifest.csv")
    current_pipeline_code_sha256 = None
    try:
        current_pipeline_code_sha256, pipeline_code_error = (
            _check_current_pipeline_code(run_config)
        )
        if pipeline_code_error:
            errors.append(f"Pipeline code validation failed: {pipeline_code_error}")
    except Exception as error:
        errors.append(
            "Pipeline code validation failed: "
            f"{type(error).__name__}: {error}"
        )
    errors.extend(_validate_control_files(selection, manifest, run_config))
    if len(selection) != 14 or len(manifest) != 14:
        errors.append(f"Expected 14 selection/manifest rows, got {len(selection)}/{len(manifest)}")
    labels = Counter(row["label"] for row in manifest)
    if labels != {"Normal": 7, "Tumor": 7}:
        errors.append(f"Expected balanced labels, got {dict(labels)}")
    if len({row["patient_id"] for row in manifest}) != len(manifest):
        errors.append("Patient IDs are not unique")
    if len({str(Path(row["slide_path"]).resolve()) for row in manifest}) != len(manifest):
        errors.append("Resolved source WSI paths are not unique")
    if any(row["status"] != "success" for row in manifest):
        errors.append("Manifest contains unsuccessful rows")

    for row in manifest:
        try:
            details.append(
                _verify_slide(
                    row,
                    output_root,
                    run_config,
                    verify_source_md5=verify_source_md5,
                    source_sample_count=source_sample_count,
                )
            )
        except Exception as error:
            errors.append(f"{row.get('slide_id', '<unknown>')}: {type(error).__name__}: {error}")

    stale = [
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.name.endswith(".tmp") or ".tmp." in path.name or ".working." in path.name or ".backup." in path.name
    ]
    if stale:
        errors.append(f"Stale temporary artifacts: {stale}")
    failure_files = sorted(str(path.relative_to(output_root)) for path in (output_root / "failures").glob("*.json")) if (output_root / "failures").exists() else []
    if failure_files:
        errors.append(f"Failure records remain: {failure_files}")

    report = {
        "status": "PASSED" if not errors else "FAILED",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "run_fingerprint": run_config.get("run_fingerprint"),
        "recorded_pipeline_code_sha256": run_config.get("pipeline_code_sha256"),
        "current_pipeline_code_sha256": current_pipeline_code_sha256,
        "verify_source_md5": bool(verify_source_md5),
        "source_sample_count_per_slide": int(source_sample_count),
        "labels": dict(labels),
        "slides": len(details),
        "patches_total": sum(item["patch_count"] for item in details),
        "patches_by_label": {
            label: sum(item["patch_count"] for item in details if item["label"] == label)
            for label in ("Normal", "Tumor")
        },
        "hdf5_bytes_total": sum(item["hdf5_bytes"] for item in details),
        "errors": errors,
        "slide_details": details,
        "control_file_sha256": {
            name: _hash_file(output_root / name)
            for name in ("run_config.json", "selection.csv", "manifest.csv")
        },
    }
    _write_text_atomic(output_root / "verification.json", json.dumps(report, indent=2) + "\n")
    lines = [
        "# Pilot-14 verification",
        "",
        f"- Status: **{report['status']}**",
        f"- Slides: {report['slides']} (Normal {labels.get('Normal', 0)}, Tumor {labels.get('Tumor', 0)})",
        f"- Valid patches: {report['patches_total']} (Normal {report['patches_by_label']['Normal']}, Tumor {report['patches_by_label']['Tumor']})",
        f"- HDF5 bytes: {report['hdf5_bytes_total']}",
        f"- Source re-extraction samples per slide: {source_sample_count}",
        f"- Full source MD5 verification: {bool(verify_source_md5)}",
        "",
    ]
    if errors:
        lines.extend(["## Errors", ""] + [f"- {error}" for error in errors] + [""])
    lines.extend([
        "## Per-slide results",
        "",
        "| Slide | Label | Patches | HDF5 bytes | Source samples |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    lines.extend(
        f"| {item['slide_id']} | {item['label']} | {item['patch_count']} | {item['hdf5_bytes']} | {item['source_reextract_samples']} |"
        for item in details
    )
    lines.append("")
    _write_text_atomic(output_root / "verification.md", "\n".join(lines))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Completed pilot output directory.")
    parser.add_argument("--verify-source-md5", action="store_true")
    parser.add_argument("--source-sample-count", type=int, default=3)
    args = parser.parse_args()
    if args.source_sample_count < 0:
        parser.error("--source-sample-count must be >= 0")
    return args


def main():
    args = parse_args()
    try:
        report = verify(
            args.output_dir,
            verify_source_md5=args.verify_source_md5,
            source_sample_count=args.source_sample_count,
        )
    except Exception as error:
        print(f"Verification setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("status", "slides", "patches_total", "patches_by_label", "hdf5_bytes_total", "errors")}, indent=2))
    return 0 if report["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
