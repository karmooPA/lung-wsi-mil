#!/usr/bin/env python3
"""Independently verify every slide in a completed WSI dataset run."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from lung_wsi_mil.paths import PROJECT_ROOT
from lung_wsi_mil.workflows import dataset as runner
from lung_wsi_mil.workflows import verify_pilot as scientific


def _validate_control_files(selection, manifest, run_config):
    """Validate dataset coverage without imposing the pilot's cohort rules."""
    errors = []
    selection_ids = [scientific._text(row.get("slide_id")).strip() for row in selection]
    manifest_ids = [scientific._text(row.get("slide_id")).strip() for row in manifest]
    if not selection_ids:
        errors.append("Selection is empty")
    if not all(selection_ids) or len(set(selection_ids)) != len(selection_ids):
        errors.append("Selection slide IDs are empty or duplicated")
    if not all(manifest_ids) or len(set(manifest_ids)) != len(manifest_ids):
        errors.append("Manifest slide IDs are empty or duplicated")
    if selection_ids != manifest_ids:
        errors.append("Selection and manifest slide IDs/order differ")
    if run_config.get("selected_slide_ids") != selection_ids:
        errors.append("run_config selected_slide_ids does not match selection.csv")
    try:
        identities = scientific._source_identity_map(run_config)
    except ValueError as error:
        errors.append(str(error))
        identities = {}
    if set(identities) != set(selection_ids):
        errors.append("run_config source identities do not match selected slide IDs")
    slide_fingerprints = run_config.get("slide_fingerprints")
    if not isinstance(slide_fingerprints, dict):
        errors.append("run_config slide_fingerprints must be an object")
        slide_fingerprints = {}
    if set(slide_fingerprints) != set(selection_ids):
        errors.append("run_config slide fingerprints do not match selected slide IDs")

    manifest_by_id = {row.get("slide_id"): row for row in manifest}
    source_paths = []
    for selected in selection:
        slide_id = scientific._text(selected.get("slide_id")).strip()
        identity = identities.get(slide_id)
        row = manifest_by_id.get(slide_id)
        if selected.get("label") not in {"Normal", "Tumor"}:
            errors.append(f"Invalid selection label for {slide_id}")
        if not scientific._text(selected.get("patient_id")).strip():
            errors.append(f"Missing patient_id for {slide_id}")
        selected_path = scientific._text(selected.get("resolved_path")).strip()
        if not selected_path:
            errors.append(f"Missing resolved source path for {slide_id}")
        else:
            source_paths.append(scientific._resolved_text(selected_path))
        if identity is None or row is None:
            continue
        try:
            scientific._compare_source_identities(selected, identity, f"selection {slide_id}")
        except ValueError as error:
            errors.append(str(error))
        for field in ("patient_id", "label"):
            if scientific._text(row.get(field)) != scientific._text(selected.get(field)):
                errors.append(f"Selection/manifest mismatch for {slide_id} {field}")
        if scientific._resolved_text(row.get("slide_path", "")) != scientific._resolved_text(
            selected_path
        ):
            errors.append(f"Selection/manifest path mismatch for {slide_id}")
        try:
            if int(row.get("source_size_bytes", -1)) != int(selected.get("source_size_bytes", -2)):
                errors.append(f"Selection/manifest source size mismatch for {slide_id}")
        except (TypeError, ValueError):
            errors.append(f"Selection/manifest source size is invalid for {slide_id}")
        expected_md5 = (
            scientific._text(selected.get("computed_md5")).lower()
            or scientific._text(selected.get("expected_md5")).lower()
        )
        if scientific._text(row.get("source_md5")).lower() != expected_md5:
            errors.append(f"Selection/manifest source MD5 mismatch for {slide_id}")
        if scientific._text(row.get("run_fingerprint")) != slide_fingerprints.get(slide_id):
            errors.append(f"Manifest slide fingerprint mismatch for {slide_id}")
    if len(set(source_paths)) != len(source_paths):
        errors.append("Resolved source WSI paths are not unique")
    return errors


def _validate_fingerprints(run_config):
    """Recompute both the cohort identity and each independently resumable slide."""
    errors = []
    settings = run_config.get("base_settings")
    if not isinstance(settings, dict):
        raise ValueError("run_config base_settings must be an object")
    for field, value in settings.items():
        if run_config.get(field) != value:
            errors.append(f"run_config {field} differs from fingerprinted base_settings")

    def fingerprint(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    expected_run = fingerprint(
        {"settings": settings, "selected_slide_ids": run_config.get("selected_slide_ids")}
    )
    expected_slides = {
        slide_id: fingerprint({"settings": settings, "source_identity": identity})
        for slide_id, identity in scientific._source_identity_map(run_config).items()
    }
    if run_config.get("run_fingerprint") != expected_run:
        errors.append("run_config fingerprint does not match its contents")
    observed_slides = run_config.get("slide_fingerprints")
    if observed_slides != expected_slides:
        errors.append("run_config slide fingerprints do not match their source/settings")
    return errors


def _validate_dataset_coverage(selection, run_config):
    dataset_root = Path(run_config["dataset_root"]).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    observed = {
        path.resolve()
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in runner.pilot.SUPPORTED_EXTENSIONS
    }
    selected = {Path(row["resolved_path"]).expanduser().resolve() for row in selection}
    errors = []
    if observed != selected:
        missing = sorted(str(path) for path in observed - selected)
        absent = sorted(str(path) for path in selected - observed)
        errors.append(
            f"Dataset/selection WSI coverage differs: unselected={missing}; absent={absent}"
        )
    if int(run_config.get("local_wsi_files_discovered", -1)) != len(observed):
        errors.append("run_config local_wsi_files_discovered differs from the current dataset")
    return errors


def _initialize_worker(threads):
    scientific.torch.set_num_threads(threads)
    try:
        scientific.torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def _verify_one(row, output_root, run_config, verify_source_md5, source_sample_count):
    slide_id = scientific._text(row.get("slide_id")) or "<unknown>"
    try:
        if row.get("status") != "success":
            raise ValueError(
                f"Manifest status is {row.get('status')!r}: {row.get('error_message', '')}"
            )
        slide_config = dict(run_config)
        slide_config["run_fingerprint"] = run_config["slide_fingerprints"][slide_id]
        detail = scientific._verify_slide(
            row,
            Path(output_root),
            slide_config,
            verify_source_md5=verify_source_md5,
            source_sample_count=source_sample_count,
        )
        return detail, None
    except Exception as error:
        return None, {
            "slide_id": slide_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def _write_report(output_root, report):
    scientific._write_text_atomic(
        output_root / "verification.json", json.dumps(report, indent=2) + "\n"
    )
    labels = report["labels"]
    lines = [
        "# WSI dataset verification",
        "",
        f"- Status: **{report['status']}**",
        f"- Verified slides: {report['slides']} / {report['selected_slides']} (Normal {labels.get('Normal', 0)}, Tumor {labels.get('Tumor', 0)})",
        f"- Failed slide checks: {len(report['failed_slide_details'])}",
        f"- Valid patches: {report['patches_total']} (Normal {report['patches_by_label']['Normal']}, Tumor {report['patches_by_label']['Tumor']})",
        f"- HDF5 bytes: {report['hdf5_bytes_total']}",
        f"- Source re-extraction samples per slide: {report['source_sample_count_per_slide']}",
        f"- Full source MD5 verification: {report['verify_source_md5']}",
        "",
    ]
    if report["errors"]:
        lines.extend(["## Errors", ""] + [f"- {error}" for error in report["errors"]] + [""])
    lines.extend(
        [
            "## Per-slide results",
            "",
            "| Slide | Label | Patches | HDF5 bytes | Source samples |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {item['slide_id']} | {item['label']} | {item['patch_count']} | {item['hdf5_bytes']} | {item['source_reextract_samples']} |"
        for item in report["slide_details"]
    )
    lines.append("")
    scientific._write_text_atomic(output_root / "verification.md", "\n".join(lines))


def verify(output_dir, verify_source_md5=False, source_sample_count=3, workers=1, threads=1):
    if source_sample_count < 0 or workers < 1 or threads < 1:
        raise ValueError(
            "Source sample count must be nonnegative; workers and threads must be positive"
        )
    output_root = Path(output_dir).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_root}")
    errors, details, failed_details = [], [], []
    controls = {}
    for name in ("run_config.json", "selection.csv", "manifest.csv"):
        try:
            path = output_root / name
            payload = (
                json.loads(path.read_text(encoding="utf-8"))
                if name.endswith(".json")
                else scientific._read_csv(path)
            )
            if name.endswith(".json") and not isinstance(payload, dict):
                raise ValueError("run_config must be an object")
            controls[name] = payload
        except Exception as error:
            errors.append(f"Cannot read {name}: {type(error).__name__}: {error}")
    run_config = controls.get("run_config.json", {})
    selection = controls.get("selection.csv", [])
    manifest = controls.get("manifest.csv", [])
    control_hashes = {name: scientific._hash_file(output_root / name) for name in controls}
    current_code_hash = None
    if len(controls) == 3:
        try:
            current_code_hash = runner._dataset_code_fingerprint(PROJECT_ROOT)
            if run_config.get("pipeline_code_sha256") != current_code_hash:
                errors.append(
                    "Current pipeline code SHA256 does not match the code that generated the outputs"
                )
            errors.extend(_validate_control_files(selection, manifest, run_config))
            errors.extend(_validate_fingerprints(run_config))
            errors.extend(_validate_dataset_coverage(selection, run_config))
        except Exception as error:
            errors.append(f"Control validation failed: {type(error).__name__}: {error}")

        _initialize_worker(threads)
        tasks = [
            (row, str(output_root), run_config, bool(verify_source_md5), int(source_sample_count))
            for row in manifest
        ]
        if workers == 1:
            results = (_verify_one(*task) for task in tasks)
            for index, (detail, failure) in enumerate(results, 1):
                if detail:
                    details.append(detail)
                if failure:
                    failed_details.append(failure)
                print(
                    f"Verified {index}/{len(tasks)} slides; failures {len(failed_details)}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_initialize_worker, initargs=(threads,)
            ) as pool:
                futures = {
                    pool.submit(_verify_one, *task): task[0].get("slide_id", "<unknown>")
                    for task in tasks
                }
                for index, future in enumerate(as_completed(futures), 1):
                    try:
                        detail, failure = future.result()
                    except Exception as error:
                        detail = None
                        failure = {
                            "slide_id": futures[future],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    if detail:
                        details.append(detail)
                    if failure:
                        failed_details.append(failure)
                    print(
                        f"Verified {index}/{len(tasks)} slides; failures {len(failed_details)}",
                        flush=True,
                    )

    positions = {row.get("slide_id"): index for index, row in enumerate(manifest)}
    details.sort(key=lambda item: positions.get(item["slide_id"], len(positions)))
    failed_details.sort(key=lambda item: positions.get(item["slide_id"], len(positions)))
    errors.extend(
        f"{item['slide_id']}: {item['error_type']}: {item['error_message']}"
        for item in failed_details
    )
    stale = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.name.endswith(".tmp")
        or any(token in path.name for token in (".tmp.", ".working.", ".backup."))
    )
    if stale:
        errors.append(f"Stale temporary artifacts: {stale}")
    failures = sorted(
        str(path.relative_to(output_root)) for path in (output_root / "failures").glob("*.json")
    )
    if failures:
        errors.append(f"Failure records remain: {failures}")
    for name, digest in control_hashes.items():
        try:
            if scientific._hash_file(output_root / name) != digest:
                errors.append(f"Control file changed during verification: {name}")
        except OSError as error:
            errors.append(f"Control file became unreadable during verification: {name}: {error}")
    labels = Counter(row.get("label", "<missing>") for row in manifest)
    report = {
        "status": "FAILED" if errors else "PASSED",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "run_fingerprint": run_config.get("run_fingerprint"),
        "recorded_pipeline_code_sha256": run_config.get("pipeline_code_sha256"),
        "current_pipeline_code_sha256": current_code_hash,
        "verify_source_md5": bool(verify_source_md5),
        "source_sample_count_per_slide": int(source_sample_count),
        "workers": workers,
        "threads_per_worker": threads,
        "selected_slides": len(selection),
        "manifest_slides": len(manifest),
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
        "failed_slide_details": failed_details,
        "control_file_sha256": control_hashes,
    }
    _write_report(output_root, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Completed dataset output directory.")
    parser.add_argument("--verify-source-md5", action="store_true")
    parser.add_argument("--source-sample-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.source_sample_count < 0 or args.workers < 1 or args.threads < 1:
        parser.error("--source-sample-count must be >= 0; --workers and --threads must be >= 1")
    return args


def main():
    args = parse_args()
    try:
        report = verify(
            args.output_dir,
            args.verify_source_md5,
            args.source_sample_count,
            args.workers,
            args.threads,
        )
    except Exception as error:
        print(f"Verification setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "slides",
                    "patches_total",
                    "patches_by_label",
                    "hdf5_bytes_total",
                    "errors",
                )
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
