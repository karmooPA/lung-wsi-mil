#!/usr/bin/env python3
"""Run the existing preprocessing/augmentation pipeline on every local WSI.

This orchestrator preserves the pilot's scientific operators and per-slide
artifact format. Cohort membership is not part of a slide's resume fingerprint.
"""

import argparse
import errno
import hashlib
import math
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from lung_wsi_mil.paths import PROJECT_ROOT
from lung_wsi_mil.workflows import pilot as pilot

DATASET_VERSION = "1.0.0"
GIB = 1024**3
BYTES_PER_PATCH = len(pilot.REPRESENTATIONS) * pilot.PATCH_SIZE**2 * 3
IDENTITY_FIELDS = (
    "slide_id",
    "patient_id",
    "label",
    "resolved_path",
    "source_size_bytes",
    "source_mtime_ns",
    "expected_md5",
    "computed_md5",
)


def _now():
    return datetime.now(UTC).isoformat()


def _select_all_present(inventory_rows, dataset_root, local_paths, by_name, by_slide_prefix):
    """Require complete, unambiguous physical coverage; patients may repeat."""
    selected = []
    known_paths = set()
    slide_ids = set()
    directory_names = set()
    discovered = {Path(path).resolve() for path in local_paths}
    if not discovered:
        raise ValueError(f"No supported WSI files found under {dataset_root}.")
    for original in inventory_rows:
        path, method = pilot._resolve_inventory_row(
            original, dataset_root, by_name, by_slide_prefix
        )
        if path is None:
            continue
        path = Path(path).resolve()
        if path not in discovered:
            raise ValueError(f"Inventory resolves to a non-supported WSI: {path}")
        row = dict(original)
        slide_id = str(row.get("slide_id") or "").strip()
        patient_id = str(row.get("patient_id") or "").strip()
        label = str(row.get("label") or "").strip()
        if not slide_id:
            raise ValueError(f"Present WSI has no slide_id: {path}")
        if not patient_id or patient_id.upper() in pilot.NA_VALUES:
            raise ValueError(f"Present WSI has no valid patient_id: {slide_id}")
        if label not in pilot.LABELS:
            raise ValueError(f"Present WSI has an unsupported label: {slide_id}: {label!r}")
        if pilot._parse_bool(row.get("duplicate")) is True:
            raise ValueError(f"Inventory marks a present WSI as duplicate: {slide_id}")
        if pilot._parse_bool(row.get("label_valid")) is False:
            raise ValueError(f"Inventory marks a present WSI label invalid: {slide_id}")
        if slide_id in slide_ids:
            raise ValueError(f"Duplicate present slide_id: {slide_id}")
        if path in known_paths:
            raise ValueError(f"Duplicate physical WSI in inventory: {path}")
        directory_name = pilot._safe_slide_dir_name(slide_id)
        # Case-folded names also protect Windows case-insensitive filesystems.
        if directory_name.casefold() in directory_names:
            raise ValueError(f"Output directory name collision for slide_id: {slide_id}")
        slide_ids.add(slide_id)
        known_paths.add(path)
        directory_names.add(directory_name.casefold())
        stat = path.stat()
        row.update(
            slide_id=slide_id,
            patient_id=patient_id,
            label=label,
            resolved_path=str(path),
            path_resolution_method=method,
            source_size_bytes=int(stat.st_size),
            source_mtime_ns=int(stat.st_mtime_ns),
            expected_md5=str(row.get("expected_md5") or "").strip().lower(),
            computed_md5="",
        )
        selected.append(row)
    uncovered = discovered.difference(known_paths)
    if uncovered:
        examples = "; ".join(str(path) for path in sorted(uncovered)[:10])
        raise ValueError(
            f"Inventory does not unambiguously cover {len(uncovered)} local WSI file(s): {examples}"
        )
    return sorted(selected, key=lambda row: (row["label"], row["slide_id"]))


def _source_identity(row):
    return {field: row.get(field, "") for field in IDENTITY_FIELDS}


def _dataset_code_fingerprint(project_root=None):
    digest = hashlib.sha256()
    digest.update(pilot._code_fingerprint(project_root).encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _slide_fingerprint(base_settings, source_identity):
    return pilot._run_fingerprint({"settings": base_settings, "source_identity": source_identity})


def _verification_fingerprints(run_config):
    """Recompute recorded digests for the independent dataset verifier."""
    settings = run_config["base_settings"]
    for key, value in settings.items():
        if run_config.get(key) != value:
            raise ValueError(f"run_config setting differs from base_settings: {key}")
    overall = pilot._run_fingerprint(
        {"settings": settings, "selected_slide_ids": run_config["selected_slide_ids"]}
    )
    slides = {
        identity["slide_id"]: _slide_fingerprint(settings, _source_identity(identity))
        for identity in run_config["source_identities"]
    }
    return overall, slides


def _prevent_sleep():
    """Temporarily keep Windows awake without changing its power settings."""
    if os.name != "nt":
        return False
    import ctypes

    return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))


def _restore_sleep():
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def _ensure_disk_space(output_root, *, min_free_gib, patch_count=0, workers=1):
    # Compression may help, but admission uses the uncompressed upper bound.
    # Multiplying by workers reserves room for concurrently admitted slides.
    required = math.ceil(min_free_gib * GIB) + int(patch_count) * BYTES_PER_PATCH * int(workers)
    free = shutil.disk_usage(output_root).free
    if free < required:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient output disk space: {free / GIB:.2f} GiB free; "
            f"{required / GIB:.2f} GiB required including reserve.",
        )
    return free


def _manifest_row(row):
    return {
        "slide_id": row["slide_id"],
        "patient_id": row["patient_id"],
        "label": row["label"],
        "status": "pending",
        "slide_path": row["resolved_path"],
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
        "run_fingerprint": "",
        "resumed": False,
    }


def _acquire_lock(output_root):
    path = Path(output_root) / ".dataset.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(
            f"A dataset run may already be active ({path}; {owner}). "
            "Remove this lock only after confirming the recorded process is inactive."
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted_at={_now()}\n")
    return path


def _initialize_worker():
    # Several process workers with large torch thread pools oversubscribe CPUs.
    pilot.torch.set_num_threads(1)


def _process_slide(row, output_root, options, base_settings, preprocessing_config):
    """Own one slide directory; parent alone owns cohort control files."""
    output_root = Path(output_root)
    slide_id = row["slide_id"]
    slide_path = Path(row["resolved_path"])
    safe_name = pilot._safe_slide_dir_name(slide_id)
    slide_dir = output_root / "slides" / safe_name
    working_dir = None
    owns_working_dir = False
    started_total = time.perf_counter()
    manifest_row = _manifest_row(row)
    identity = _source_identity(row)
    fingerprint = ""
    try:
        _ensure_disk_space(output_root, min_free_gib=options["min_free_gib"])
        # Source validation is per-slide, so one corrupt source cannot abort the cohort.
        prepared = pilot._prepare_source_identities([row], verify_md5=options["verify_source_md5"])[
            0
        ]
        if (prepared["source_size_bytes"], prepared["source_mtime_ns"]) != (
            row["source_size_bytes"],
            row["source_mtime_ns"],
        ):
            raise ValueError(f"Source changed after selection preflight: {slide_id}")
        identity = _source_identity(prepared)
        fingerprint = _slide_fingerprint(base_settings, identity)
        manifest_row.update(
            run_fingerprint=fingerprint,
            source_md5=prepared.get("computed_md5") or prepared.get("expected_md5", ""),
        )
        print(f"START {slide_id} | {row['label']}", flush=True)
        if not options["force"]:
            completed = pilot._completed_payload(
                slide_dir,
                fingerprint,
                options["save_qc"],
                preprocessing_config=preprocessing_config,
                source_identity=identity,
                qc_sample_count=options["qc_sample_count"],
            )
            if completed is not None:
                manifest_row = dict(completed["manifest_row"])
                # Keep the completion marker and cohort manifest fields identical.
                (output_root / "failures" / f"{safe_name}.json").unlink(missing_ok=True)
                print(f"RESUME {slide_id}: validated existing result", flush=True)
                return {
                    "manifest_row": manifest_row,
                    "source_identity": identity,
                    "run_fingerprint": fingerprint,
                    "resumed": True,
                }

        working_dir = output_root / "slides" / f".{safe_name}.working.{os.getpid()}"
        if working_dir.exists():
            raise RuntimeError(f"Refusing to overwrite a stale working directory: {working_dir}")
        working_dir.mkdir(parents=True, exist_ok=False)
        owns_working_dir = True
        started = time.perf_counter()
        result = pilot.preprocess_slide(
            slide_path=slide_path,
            target_magnification=options["target_magnification"],
            patch_size=pilot.PATCH_SIZE,
            min_tissue_percentage=options["min_tissue_percentage"],
            color_distance_threshold=options["color_distance_threshold"],
            thumbnail_max_size=options["thumbnail_max_size"],
        )
        preprocessing_seconds = time.perf_counter() - started
        pilot.save_slide_outputs(
            result=result,
            slide_path=slide_path,
            output_dir=working_dir / "preprocessing",
            config=preprocessing_config,
            elapsed_seconds=preprocessing_seconds,
            save_qc=options["save_qc"],
        )
        _ensure_disk_space(
            output_root,
            min_free_gib=options["min_free_gib"],
            patch_count=result["valid_patch_count"],
            workers=options["workers"],
        )
        started = time.perf_counter()
        augmentation = pilot._save_augmented_views(
            slide_path=slide_path,
            slide_id=slide_id,
            patient_id=row["patient_id"],
            label=row["label"],
            result=result,
            output_path=working_dir / "augmentation" / "patch_views.h5",
            master_seed=options["seed"],
            chunk_size=options["chunk_size"],
            save_qc=options["save_qc"],
            qc_sample_count=options["qc_sample_count"],
            source_identity=identity,
            run_fingerprint=fingerprint,
        )
        augmentation_seconds = time.perf_counter() - started
        source_stat = slide_path.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (
            identity["source_size_bytes"],
            identity["source_mtime_ns"],
        ):
            raise ValueError(f"Source changed during processing: {slide_id}")
        manifest_row.update(
            status="success",
            candidate_tile_count=result["candidate_tile_count"],
            valid_patch_count=result["valid_patch_count"],
            feature_extraction_ready=result["feature_extraction_ready"],
            augmentation_hdf5=str(slide_dir / "augmentation" / "patch_views.h5"),
            augmentation_size_bytes=augmentation["size_bytes"],
            augmentation_sha256=augmentation["sha256"],
            augmentation_qc_montage=(
                str(slide_dir / "augmentation" / "augmentation_qc_montage.png")
                if augmentation.get("qc_montage")
                else ""
            ),
            augmentation_quality_csv=augmentation["quality_csv"],
            preprocessing_seconds=f"{preprocessing_seconds:.6f}",
            augmentation_seconds=f"{augmentation_seconds:.6f}",
            total_seconds=f"{time.perf_counter() - started_total:.6f}",
        )
        pilot._write_json_atomic(
            {
                "status": "success",
                "run_fingerprint": fingerprint,
                "slide_seed": augmentation["slide_seed"],
                "source_identity": identity,
                "manifest_row": manifest_row,
            },
            working_dir / "complete.json",
        )
        if (
            pilot._completed_payload(
                working_dir,
                fingerprint,
                options["save_qc"],
                preprocessing_config=preprocessing_config,
                source_identity=identity,
                qc_sample_count=options["qc_sample_count"],
            )
            is None
        ):
            raise RuntimeError("New slide artifacts failed post-write integrity validation.")
        pilot._install_slide_directory(working_dir, slide_dir)
        working_dir = None
        (output_root / "failures" / f"{safe_name}.json").unlink(missing_ok=True)
        print(
            f"SAVED {slide_id}: {result['valid_patch_count']} patches, "
            f"{augmentation['size_bytes'] / GIB:.3f} GiB",
            flush=True,
        )
        return {
            "manifest_row": manifest_row,
            "source_identity": identity,
            "run_fingerprint": fingerprint,
            "resumed": False,
        }
    except Exception as error:
        manifest_row.update(
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
            total_seconds=f"{time.perf_counter() - started_total:.6f}",
        )
        failure = {
            "status": "failed",
            "slide_id": slide_id,
            "run_fingerprint": fingerprint,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            pilot._write_json_atomic(failure, output_root / "failures" / f"{safe_name}.json")
        except OSError:
            pass  # The parent manifest still records failures if the disk fills.
        if owns_working_dir and working_dir is not None and working_dir.exists():
            shutil.rmtree(working_dir)
        print(f"FAILED {slide_id}: {type(error).__name__}: {error}", flush=True)
        return {
            "manifest_row": manifest_row,
            "source_identity": identity,
            "run_fingerprint": fingerprint,
            "resumed": False,
        }


def _write_control_files(
    output_root, selected, manifest, run_config, active, state, started_at, resumed
):
    pilot._save_selection(selected, output_root)
    pilot._write_csv_atomic(
        output_root / "manifest.csv",
        pilot._manifest_fieldnames() + ["run_fingerprint", "resumed"],
        manifest,
    )
    pilot._write_json_atomic(run_config, output_root / "run_config.json")
    success = sum(row["status"] == "success" for row in manifest)
    failed = sum(row["status"] == "failed" for row in manifest)
    status = {
        "state": state,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": _now(),
        "total": len(manifest),
        "success": success,
        "successes": success,
        "completed": success + failed,
        "failed": failed,
        "resumed": resumed,
        "running": len(active),
        "inflight": len(active),
        "pending": len(manifest) - success - failed - len(active),
        "active_slide_ids": sorted(active),
        "workers": run_config["workers"],
        "free_disk_bytes": shutil.disk_usage(output_root).free,
        "manifest": str(output_root / "manifest.csv"),
    }
    pilot._write_json_atomic(status, output_root / "status.json")


def run(args):
    output_root = Path(args.output_dir).expanduser().resolve()
    project_root = PROJECT_ROOT
    pilot._validate_run_directories(args.dataset_root, output_root, project_root)
    dataset_root, local_paths, by_name, by_prefix = pilot._index_local_slides(args.dataset_root)
    inventory_path = Path(args.inventory).expanduser().resolve()
    selected = _select_all_present(
        pilot._load_inventory(inventory_path), dataset_root, local_paths, by_name, by_prefix
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(output_root)
    sleep_prevented = False
    try:
        sleep_prevented = _prevent_sleep()
        if os.name == "nt" and not sleep_prevented:
            print("Windows sleep prevention request was unavailable.", file=sys.stderr, flush=True)
        return _run_locked(args, selected, dataset_root, output_root, inventory_path, project_root)
    finally:
        if sleep_prevented:
            _restore_sleep()
        lock.unlink(missing_ok=True)


def _run_locked(args, selected, dataset_root, output_root, inventory_path, project_root):
    preprocessing_config = pilot.build_config(
        target_magnification=args.target_magnification,
        patch_size=pilot.PATCH_SIZE,
        min_tissue_percentage=args.min_tissue_percentage,
        color_distance_threshold=args.color_distance_threshold,
        thumbnail_max_size=args.thumbnail_max_size,
        save_qc=args.save_qc,
    )
    base_settings = {
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "save_qc": args.save_qc,
        "qc_sample_count": args.qc_sample_count,
        "verify_source_md5": args.verify_source_md5,
        "preprocessing": preprocessing_config,
        "pipeline_code_sha256": _dataset_code_fingerprint(project_root),
        "runtime": pilot._runtime_versions(),
    }
    identities = [_source_identity(row) for row in selected]
    run_config = {
        **base_settings,
        "base_settings": base_settings,
        "dataset_root": str(dataset_root),
        "inventory": str(inventory_path),
        "inventory_sha256": pilot._hash_file(inventory_path),
        "selection_rule": "all local supported WSIs; unique slide IDs and physical paths",
        "selected_slide_ids": [row["slide_id"] for row in selected],
        "source_identities": identities,
        "local_wsi_files_discovered": len(selected),
        "slide_fingerprints": {},
        "workers": args.workers,
        "min_free_gib": args.min_free_gib,
        "preprocessing_fingerprint": pilot.config_fingerprint(preprocessing_config),
    }
    # This is a cohort identifier. HDF5/completion markers use slide_fingerprints.
    run_config["run_fingerprint"] = pilot._run_fingerprint(
        {"settings": base_settings, "selected_slide_ids": run_config["selected_slide_ids"]}
    )
    options = vars(args).copy()
    manifest = [_manifest_row(row) for row in selected]
    indices = {row["slide_id"]: index for index, row in enumerate(selected)}
    active = set()
    started_at = _now()
    resumed = 0
    state = "running"
    _write_control_files(
        output_root, selected, manifest, run_config, active, state, started_at, resumed
    )
    print(
        f"Dataset: {dataset_root}\nWSIs: {len(selected)}\nOutput: {output_root}\nWorkers: {args.workers}",
        flush=True,
    )

    def accept(result):
        nonlocal resumed
        row = result["manifest_row"]
        index = indices[row["slide_id"]]
        manifest[index] = row
        identities[index] = result["source_identity"]
        selected[index].update(result["source_identity"])
        if result["run_fingerprint"]:
            run_config["slide_fingerprints"][row["slide_id"]] = result["run_fingerprint"]
        resumed += int(result["resumed"])
        active.discard(row["slide_id"])
        _write_control_files(
            output_root, selected, manifest, run_config, active, state, started_at, resumed
        )
        completed = sum(item["status"] in {"success", "failed"} for item in manifest)
        print(
            f"PROGRESS {completed}/{len(selected)}: "
            f"{sum(item['status'] == 'success' for item in manifest)} success, "
            f"{sum(item['status'] == 'failed' for item in manifest)} failed",
            flush=True,
        )

    try:
        if args.workers == 1:
            _initialize_worker()
            for row in selected:
                active.add(row["slide_id"])
                _write_control_files(
                    output_root, selected, manifest, run_config, active, state, started_at, resumed
                )
                accept(
                    _process_slide(row, output_root, options, base_settings, preprocessing_config)
                )
        else:
            # Keep only workers tasks outstanding; the parent commits one manifest
            # update per result, and every slide has exactly one directory owner.
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            ) as executor:
                pending = {}
                next_index = 0
                while next_index < len(selected) or pending:
                    while next_index < len(selected) and len(pending) < args.workers:
                        row = selected[next_index]
                        future = executor.submit(
                            _process_slide,
                            row,
                            output_root,
                            options,
                            base_settings,
                            preprocessing_config,
                        )
                        pending[future] = row
                        active.add(row["slide_id"])
                        next_index += 1
                    _write_control_files(
                        output_root,
                        selected,
                        manifest,
                        run_config,
                        active,
                        state,
                        started_at,
                        resumed,
                    )
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = pending.pop(future)
                        accept(future.result())
        state = (
            "completed"
            if all(row["status"] == "success" for row in manifest)
            else "completed_with_failures"
        )
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    except Exception:
        state = "aborted"
        raise
    finally:
        active.clear()
        _write_control_files(
            output_root, selected, manifest, run_config, active, state, started_at, resumed
        )
    return 0 if state == "completed" else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_dir")
    parser.add_argument("--inventory", default="metadata/dataset_inventory.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--qc-sample-count", type=int, default=6)
    parser.add_argument("--target-magnification", type=float, default=5.0)
    parser.add_argument("--min-tissue-percentage", type=float, default=20.0)
    parser.add_argument("--color-distance-threshold", type=float, default=35.0)
    parser.add_argument("--thumbnail-max-size", type=int, default=1024)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent slide processes, each with one torch thread.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=2.0,
        help="Free disk reserve plus uncompressed in-flight image estimate.",
    )
    parser.add_argument(
        "--verify-source-md5",
        action="store_true",
        help="Hash and validate each source before processing/resume.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-qc", dest="save_qc", action="store_false")
    parser.set_defaults(save_qc=True)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    for name in ("chunk_size", "thumbnail_max_size", "workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.qc_sample_count < 0 or (args.save_qc and args.qc_sample_count == 0):
        parser.error("--qc-sample-count must be positive unless --no-qc is used")
    if not math.isfinite(args.target_magnification) or args.target_magnification <= 0:
        parser.error("--target-magnification must be finite and > 0")
    if not math.isfinite(args.min_tissue_percentage) or not 0 <= args.min_tissue_percentage <= 100:
        parser.error("--min-tissue-percentage must be between 0 and 100")
    for name in ("color_distance_threshold", "min_free_gib"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and >= 0")
    return args


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
