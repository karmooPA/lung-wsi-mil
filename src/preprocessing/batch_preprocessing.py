import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from src.preprocessing.pipeline import preprocess_slide
from src.preprocessing.qc import save_qc_images


PREPROCESSING_VERSION = "0.2.0"
DEFAULT_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _write_json_atomic(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(data),
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporary_path, path)


def discover_slides(input_path, extensions=DEFAULT_EXTENSIONS):
    input_path = Path(input_path).expanduser().resolve()
    normalized_extensions = {extension.lower() for extension in extensions}

    if input_path.is_file():
        if input_path.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Unsupported WSI extension: {input_path.suffix}"
            )
        return [input_path], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    slides = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )

    return slides, input_path


def build_config(
    target_magnification,
    patch_size,
    min_tissue_percentage,
    color_distance_threshold,
    thumbnail_max_size,
):
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "target_magnification": float(target_magnification),
        "patch_size": int(patch_size),
        "min_tissue_percentage": float(min_tissue_percentage),
        "color_distance_threshold": float(color_distance_threshold),
        "thumbnail_max_size": int(thumbnail_max_size),
        "minimum_patches_per_slide": None,
    }


def config_fingerprint(config):
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def slide_output_dir(output_root, input_root, slide_path):
    relative_path = slide_path.relative_to(input_root)
    relative_without_suffix = relative_path.with_suffix("")
    return Path(output_root) / "slides" / relative_without_suffix


def _lightweight_result(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"thumbnail", "filtered_mask"}
    }


def save_slide_outputs(
    result,
    slide_path,
    output_dir,
    config,
    elapsed_seconds,
    save_qc,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_patches = result["valid_patches"]
    coordinates = np.asarray(
        [[patch["x"], patch["y"]] for patch in valid_patches],
        dtype=np.int64,
    )

    if coordinates.size == 0:
        coordinates = np.empty((0, 2), dtype=np.int64)

    np.save(output_dir / "coordinates.npy", coordinates)

    with (output_dir / "valid_patches.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["x", "y", "tissue_percentage"],
        )
        writer.writeheader()
        writer.writerows(valid_patches)

    qc_files = None
    if save_qc:
        qc_files = save_qc_images(result, output_dir / "qc")

    payload = {
        "status": "success",
        "slide_path": str(slide_path),
        "config": config,
        "config_fingerprint": config_fingerprint(config),
        "elapsed_seconds": float(elapsed_seconds),
        "result": _lightweight_result(result),
        "files": {
            "coordinates": "coordinates.npy",
            "valid_patches": "valid_patches.csv",
            "qc": qc_files,
        },
    }

    _write_json_atomic(payload, output_dir / "result.json")

    error_path = output_dir / "error.json"
    if error_path.exists():
        error_path.unlink()

    return payload


def save_slide_error(
    slide_path,
    output_dir,
    config,
    elapsed_seconds,
    error,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "failed",
        "slide_path": str(slide_path),
        "config": config,
        "config_fingerprint": config_fingerprint(config),
        "elapsed_seconds": float(elapsed_seconds),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }

    _write_json_atomic(payload, output_dir / "error.json")
    return payload


def load_completed_result(output_dir, config, force=False):
    if force:
        return None

    result_path = Path(output_dir) / "result.json"
    if not result_path.exists():
        return None

    try:
        with result_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if payload.get("status") != "success":
        return None

    if payload.get("config_fingerprint") != config_fingerprint(config):
        return None

    return payload


def manifest_row(payload):
    status = payload.get("status", "unknown")
    row = {
        "slide_path": payload.get("slide_path"),
        "status": status,
        "candidate_tile_count": "",
        "valid_patch_count": "",
        "feature_extraction_ready": "",
        "source_magnification": "",
        "target_magnification": "",
        "target_mpp_x": "",
        "target_mpp_y": "",
        "elapsed_seconds": payload.get("elapsed_seconds", ""),
        "error_type": payload.get("error_type", ""),
        "error_message": payload.get("error_message", ""),
    }

    if status == "success":
        result = payload.get("result", {})
        resolution = result.get("resolution", {})
        row.update(
            {
                "candidate_tile_count": result.get("candidate_tile_count", ""),
                "valid_patch_count": result.get("valid_patch_count", ""),
                "feature_extraction_ready": result.get(
                    "feature_extraction_ready", ""
                ),
                "source_magnification": resolution.get(
                    "source_magnification", ""
                ),
                "target_magnification": resolution.get(
                    "target_magnification", ""
                ),
                "target_mpp_x": resolution.get("target_mpp_x", ""),
                "target_mpp_y": resolution.get("target_mpp_y", ""),
            }
        )

    return row


def write_manifest(rows, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    fieldnames = [
        "slide_path",
        "status",
        "candidate_tile_count",
        "valid_patch_count",
        "feature_extraction_ready",
        "source_magnification",
        "target_magnification",
        "target_mpp_x",
        "target_mpp_y",
        "elapsed_seconds",
        "error_type",
        "error_message",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(
    input_path,
    output_root,
    target_magnification=5,
    patch_size=512,
    min_tissue_percentage=20.0,
    color_distance_threshold=35,
    thumbnail_max_size=1024,
    save_qc=False,
    force=False,
    extensions=DEFAULT_EXTENSIONS,
):
    slides, input_root = discover_slides(input_path, extensions=extensions)

    config = build_config(
        target_magnification=target_magnification,
        patch_size=patch_size,
        min_tissue_percentage=min_tissue_percentage,
        color_distance_threshold=color_distance_threshold,
        thumbnail_max_size=thumbnail_max_size,
    )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(slides)} WSI file(s).")
    print(f"Output: {output_root}")
    print(f"Config fingerprint: {config_fingerprint(config)}")
    print("Minimum patch count per slide: DISABLED")

    rows = []
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for index, slide_path in enumerate(slides, start=1):
        output_dir = slide_output_dir(
            output_root=output_root,
            input_root=input_root,
            slide_path=slide_path,
        )

        print(f"\n[{index}/{len(slides)}] {slide_path}")

        completed = load_completed_result(
            output_dir=output_dir,
            config=config,
            force=force,
        )

        if completed is not None:
            skipped_count += 1
            success_count += 1
            rows.append(manifest_row(completed))
            write_manifest(rows, output_root)
            print(
                "Resume: already completed with the same configuration "
                f"({completed['result']['valid_patch_count']} valid patches)."
            )
            continue

        started = time.perf_counter()

        try:
            result = preprocess_slide(
                slide_path=slide_path,
                target_magnification=target_magnification,
                patch_size=patch_size,
                min_tissue_percentage=min_tissue_percentage,
                color_distance_threshold=color_distance_threshold,
                thumbnail_max_size=thumbnail_max_size,
            )

            elapsed = time.perf_counter() - started

            payload = save_slide_outputs(
                result=result,
                slide_path=slide_path,
                output_dir=output_dir,
                config=config,
                elapsed_seconds=elapsed,
                save_qc=save_qc,
            )

            success_count += 1
            rows.append(manifest_row(payload))

            print(
                f"Done: {result['candidate_tile_count']} candidates -> "
                f"{result['valid_patch_count']} valid patches "
                f"({elapsed:.1f}s)."
            )

            if result["valid_patch_count"] == 0:
                print(
                    "Warning: no valid patches. The slide was NOT discarded; "
                    "it was recorded with zero patches."
                )

        except Exception as error:
            elapsed = time.perf_counter() - started
            payload = save_slide_error(
                slide_path=slide_path,
                output_dir=output_dir,
                config=config,
                elapsed_seconds=elapsed,
                error=error,
            )

            failed_count += 1
            rows.append(manifest_row(payload))

            print(
                f"FAILED: {type(error).__name__}: {error}\n"
                "Continuing with the next WSI."
            )

        write_manifest(rows, output_root)

    print("\n=== BATCH SUMMARY ===")
    print(f"Total discovered: {len(slides)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Resumed/skipped: {skipped_count}")
    print(f"Manifest: {output_root / 'manifest.csv'}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch WSI preprocessing for the lung WSI MIL project. "
            "Low-patch slides are preserved; there is no 30-patch cutoff."
        )
    )

    parser.add_argument(
        "input_path",
        help="A WSI file or a directory containing WSI files.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where preprocessing outputs will be written.",
    )
    parser.add_argument(
        "--target-magnification",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--min-tissue-percentage",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--color-distance-threshold",
        type=float,
        default=35,
    )
    parser.add_argument(
        "--thumbnail-max-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--save-qc",
        action="store_true",
        help="Save thumbnail, tissue mask, and patch overlay for every WSI.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess slides even when a matching successful result exists.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="WSI extensions to discover recursively.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_batch(
        input_path=args.input_path,
        output_root=args.output_dir,
        target_magnification=args.target_magnification,
        patch_size=args.patch_size,
        min_tissue_percentage=args.min_tissue_percentage,
        color_distance_threshold=args.color_distance_threshold,
        thumbnail_max_size=args.thumbnail_max_size,
        save_qc=args.save_qc,
        force=args.force,
        extensions=tuple(args.extensions),
    )


if __name__ == "__main__":
    main()
