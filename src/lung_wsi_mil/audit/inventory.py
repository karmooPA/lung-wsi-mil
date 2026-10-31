#!/usr/bin/env python3
"""Read-only, reproducible TCGA WSI inventory and integrity audit.

Default invocation: lung-wsi-audit
Rebuilds the inventory from bundled sources and explicitly recorded pilot evidence.
Use --mode local --data-root PATH for a new scan. No WSI is altered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from lung_wsi_mil.paths import PROJECT_ROOT as PROJECT

DEFAULT_OUTPUT_ROOT = PROJECT.parent / f"{PROJECT.name}-outputs"
SOURCE = PROJECT / "metadata" / "source"
NA = "NA"
EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
TCGA_PATIENT = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:-|$)")
INVENTORY_FIELDS = [
    "slide_id",
    "patient_id",
    "label",
    "relative_path",
    "file_extension",
    "file_size_mb",
    "width",
    "height",
    "level_count",
    "mpp_x",
    "mpp_y",
    "objective_power",
    "readable",
    "duplicate",
    "file_id",
    "file_name",
    "expected",
    "present",
    "status",
    "label_source",
    "patient_id_source",
    "label_valid",
    "label_error",
    "expected_relative_path",
    "path_source",
    "file_size_bytes",
    "expected_file_size_bytes",
    "file_size_source",
    "size_matches",
    "expected_md5",
    "computed_md5",
    "md5_matches",
    "duplicate_reasons",
    "patient_mixed_labels",
    "metadata_source",
    "missing_wsi_metadata",
    "probe_regions_read",
    "level_dimensions",
    "level_downsamples",
    "inspection_error",
    "metadata_opened",
    "snapshot_verified",
    "observation_source",
    "observation_time_utc",
]


def read_csv(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def hash_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_filename(value):
    return bool(value) and not any(c in value for c in "/\\:\0") and value not in {".", ".."}


def label_from_metadata(source):
    """Cross-check all supplied label encodings; never infer from pixels."""
    found, errors = [], []
    encodings = {
        "class": {"normal": "Normal", "tumor": "Tumor"},
        "label": {"0": "Normal", "1": "Tumor", "normal": "Normal", "tumor": "Tumor"},
        "tissue_type": {"normal": "Normal", "tumor": "Tumor"},
        "sample_type": {"solid tissue normal": "Normal", "primary tumor": "Tumor"},
    }
    for field, options in encodings.items():
        value = source.get(field, "").strip()
        if not value:
            continue
        canonical = options.get(value.lower())
        if canonical is None:
            errors.append(f"unsupported {field}={value}")
        else:
            found.append(canonical)
    if not found:
        errors.append("no authoritative label")
    if len(set(found)) > 1:
        errors.append("conflicting metadata label encodings")
    return (NA if errors else found[0]), "; ".join(errors)


def load_expected(labels_path, manifest_path, snapshot_path=None):
    labels = read_csv(labels_path)
    manifest = read_csv(manifest_path, "\t")
    if not labels or not manifest:
        raise ValueError("Manifest and labels must be non-empty.")
    if set(manifest[0]) != {"id", "filename", "md5", "size", "state"}:
        raise ValueError("Manifest must contain id, filename, md5, size, state.")
    if not {"file_id", "file_name"}.issubset(labels[0]):
        raise ValueError("Labels must contain file_id and file_name.")
    mids, lids = [r["id"] for r in manifest], [r["file_id"] for r in labels]
    if len(mids) != len(set(mids)) or len(lids) != len(set(lids)):
        raise ValueError("Duplicate file IDs in source tables; resolve metadata ambiguity first.")
    if set(mids) != set(lids):
        raise ValueError("Manifest and label file IDs do not match.")
    by_id = {r["file_id"]: r for r in labels}
    snapshot = {}
    if snapshot_path is not None:
        raw = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        hits = raw["data"]["hits"]
        snapshot = {r["file_id"]: r for r in hits}
        if len(snapshot) != len(hits):
            raise ValueError("Duplicate file IDs in GDC snapshot.")
    result = []
    for m in manifest:
        s = by_id[m["id"]]
        try:
            valid_uuid = str(uuid.UUID(m["id"])) == m["id"].lower()
        except ValueError:
            valid_uuid = False
        if not valid_uuid:
            raise ValueError(f"Invalid file UUID: {m['id']}")
        if not safe_filename(m["filename"]) or s["file_name"] != m["filename"]:
            raise ValueError(f"Unsafe or conflicting filename: {m['id']}")
        size = int(m["size"])
        if size <= 0 or not re.fullmatch(r"[a-fA-F0-9]{32}", m["md5"]):
            raise ValueError(f"Invalid size/checksum: {m['id']}")
        if s.get("file_size_bytes") and int(s["file_size_bytes"]) != size:
            raise ValueError(f"Size differs between labels and manifest: {m['id']}")
        if s.get("md5sum") and s["md5sum"].lower() != m["md5"].lower():
            raise ValueError(f"MD5 differs between labels and manifest: {m['id']}")
        label, error = label_from_metadata(s)
        if snapshot_path is not None:
            hit = snapshot.get(m["id"])
            if hit is None:
                raise ValueError(f"File missing from GDC snapshot: {m['id']}")
            cases = hit.get("cases", [])
            if len(cases) != 1 or len(cases[0].get("samples", [])) != 1:
                raise ValueError(f"Ambiguous case/sample mapping in snapshot: {m['id']}")
            case, sample = cases[0], cases[0]["samples"][0]
            comparisons = {
                "file_name": hit["file_name"],
                "file_size_bytes": str(hit["file_size"]),
                "md5sum": hit["md5sum"],
                "case_id": case["case_id"],
                "case_submitter_id": case["submitter_id"],
                "sample_id": sample["sample_id"],
                "sample_submitter_id": sample["submitter_id"],
                "sample_type": sample["sample_type"],
                "tissue_type": sample["tissue_type"],
                "experimental_strategy": hit["experimental_strategy"],
            }
            mismatches = [k for k, value in comparisons.items() if s.get(k) != value]
            if mismatches:
                raise ValueError(f"CSV/snapshot mismatch for {m['id']}: {', '.join(mismatches)}")
            if (
                case["project"]["project_id"] != "TCGA-LUSC"
                or hit["access"] != "open"
                or hit["data_format"] != "SVS"
                or hit["experimental_strategy"] != "Tissue Slide"
            ):
                raise ValueError(f"File outside the intended TCGA-LUSC cohort: {m['id']}")
        slide_id = m["filename"].split(".", 1)[0]
        match = TCGA_PATIENT.match(slide_id)
        barcode_patient = match.group(1) if match else NA
        patient = s.get("case_submitter_id", "").strip() or barcode_patient
        patient_source = "GDC labels CSV" if s.get("case_submitter_id") else "TCGA barcode"
        if patient != NA and barcode_patient != NA and patient != barcode_patient:
            error = (error + "; patient ID disagrees with TCGA barcode").strip("; ")
        row = {field: NA for field in INVENTORY_FIELDS}
        relative = str(PurePosixPath(m["id"], m["filename"]))
        row.update(
            slide_id=slide_id,
            patient_id=patient,
            label=label,
            relative_path=relative,
            expected_relative_path=relative,
            file_extension=Path(m["filename"]).suffix.lower(),
            file_name=m["filename"],
            file_id=m["id"],
            expected=True,
            duplicate=False,
            expected_file_size_bytes=size,
            expected_md5=m["md5"].lower(),
            label_valid=not bool(error),
            label_error=error or NA,
            label_source="labels CSV cross-checked with GDC snapshot"
            if snapshot_path
            else "labels CSV (snapshot not checked)",
            patient_id_source=patient_source,
            path_source="expected GDC layout",
            duplicate_reasons="",
            patient_mixed_labels=False,
            status="not_inspected",
            probe_regions_read=0,
            metadata_opened=NA,
            snapshot_verified=bool(snapshot_path),
        )
        result.append(row)
    return sorted(result, key=lambda r: (r["file_id"], r["file_name"]))


def positive_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else NA
    except (ValueError, TypeError):
        return NA


def inspect_slide(path, opener):
    """Probe corners and centre at every pyramid level. This is not whole-image QC."""
    result = {
        key: NA
        for key in [
            "width",
            "height",
            "level_count",
            "mpp_x",
            "mpp_y",
            "objective_power",
            "level_dimensions",
            "level_downsamples",
        ]
    }
    result.update(
        readable=False,
        probe_regions_read=0,
        inspection_error=NA,
        metadata_opened=False,
        metadata_source="OpenSlide",
        missing_wsi_metadata=NA,
    )
    try:
        with opener(str(path)) as slide:
            result["metadata_opened"] = True
            result["width"], result["height"] = slide.dimensions
            result["level_count"] = slide.level_count
            if min(slide.dimensions) <= 0 or slide.level_count < 1:
                raise ValueError("Invalid slide geometry")
            props = slide.properties
            for column, key in [
                ("mpp_x", "openslide.mpp-x"),
                ("mpp_y", "openslide.mpp-y"),
                ("objective_power", "openslide.objective-power"),
            ]:
                result[column] = positive_number(props.get(key))
            result["level_dimensions"] = json.dumps(slide.level_dimensions, separators=(",", ":"))
            result["level_downsamples"] = json.dumps(slide.level_downsamples, separators=(",", ":"))
            if (
                len(slide.level_dimensions) != slide.level_count
                or len(slide.level_downsamples) != slide.level_count
            ):
                raise ValueError("Pyramid arrays do not match level_count")
            for level, ((w, h), ds) in enumerate(
                zip(slide.level_dimensions, slide.level_downsamples)
            ):
                if min(w, h) <= 0 or not math.isfinite(ds) or ds <= 0:
                    raise ValueError("Invalid pyramid level")
                pw, ph = min(64, w), min(64, h)
                positions = sorted(
                    {
                        (0, 0),
                        (w - pw, 0),
                        (0, h - ph),
                        ((w - pw) // 2, (h - ph) // 2),
                        (w - pw, h - ph),
                    }
                )
                for x, y in positions:
                    image = slide.read_region((int(x * ds), int(y * ds)), level, (pw, ph))
                    try:
                        image.load()
                    finally:
                        image.close()
                    result["probe_regions_read"] += 1
            result["readable"] = True
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    if result["metadata_opened"]:
        result["missing_wsi_metadata"] = (
            ";".join(k for k in ["mpp_x", "mpp_y", "objective_power"] if result[k] == NA) or "none"
        )
    return result


def scan_paths(root, log):
    paths, partials, skipped = [], [], []

    def walk_error(error):
        relative = Path(error.filename).relative_to(root).as_posix() if error.filename else "."
        skipped.append(f"{relative}: {error.strerror}")
        log.error("Directory scan error: %s", skipped[-1])

    for parent, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in sorted(dirs):
            if (Path(parent) / name).is_symlink():
                skipped.append((Path(parent) / name).relative_to(root).as_posix() + "/ (symlink)")
        dirs[:] = sorted(d for d in dirs if not (Path(parent) / d).is_symlink())
        for name in sorted(files):
            path = Path(parent) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped.append(rel)
                continue
            if name.lower().endswith((".partial", ".part")):
                partials.append(rel)
            elif path.suffix.lower() in EXTENSIONS:
                paths.append(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix()), sorted(partials), skipped


def match_paths(expected, root, paths):
    """Include unmatched local images and duplicate copies; never silently drop them."""
    by_name = defaultdict(list)
    for path in paths:
        by_name[path.name.casefold()].append(path)
    used, rows = set(), []
    for original in expected:
        preferred = root / original["expected_relative_path"]
        choices = [p for p in by_name[original["file_name"].casefold()] if p not in used]
        selected = preferred if preferred in choices else (choices[0] if len(choices) else None)
        row = dict(original)
        if selected:
            used.add(selected)
            row.update(
                relative_path=selected.relative_to(root).as_posix(),
                path_source="filesystem",
                present=True,
            )
        else:
            row.update(present=False, status="missing")
        rows.append(row)
    for path in paths:
        if path in used:
            continue
        candidates = [r for r in expected if r["file_name"].casefold() == path.name.casefold()]
        if len(candidates) == 1:
            row = dict(candidates[0])
        else:
            row = {f: NA for f in INVENTORY_FIELDS}
            name = path.name.split(".", 1)[0]
            match = TCGA_PATIENT.match(name)
            row.update(
                slide_id=name,
                patient_id=match.group(1) if match else NA,
                label=NA,
                label_valid=False,
                label_error="No unique authoritative label match",
                patient_id_source="TCGA barcode" if match else NA,
                file_name=path.name,
                duplicate=False,
                duplicate_reasons="",
                patient_mixed_labels=False,
            )
        row.update(
            relative_path=path.relative_to(root).as_posix(),
            file_extension=path.suffix.lower(),
            path_source="filesystem",
            expected=False,
            present=True,
            status="unexpected_file",
        )
        rows.append(row)
    return rows


def audit_local_rows(rows, root, opener, verify_md5, log):
    todo = [r for r in rows if r["present"] is True]
    for i, row in enumerate(todo, 1):
        path = root / row["relative_path"]
        log.info("Scanning slides: %s / %s", i, len(todo))
        try:
            before = path.stat()
            row.update(
                file_size_bytes=before.st_size,
                file_size_mb=round(before.st_size / 10**6, 6),
                file_size_source="filesystem",
            )
            expected_size = row["expected_file_size_bytes"]
            row["size_matches"] = before.st_size == expected_size if expected_size != NA else NA
            if row["size_matches"] is False:
                row.update(
                    status="size_mismatch",
                    readable=NA,
                    inspection_error="Size differs from manifest; possible unfinished/corrupt download",
                )
                log.warning("Size mismatch: %s", row["relative_path"])
                continue
            row.update(inspect_slide(path, opener))
            row.update(
                observation_source="current_run", observation_time_utc=datetime.now(UTC).isoformat()
            )
            if row["readable"] is False:
                log.error(
                    "%s - cannot open slide or decode sampled regions: %s",
                    row["relative_path"],
                    row["inspection_error"],
                )
            if verify_md5:
                log.info("Computing MD5: %s", row["file_name"])
                row["computed_md5"] = hash_file(path, "md5")
                row["md5_matches"] = (
                    row["computed_md5"] == row["expected_md5"] if row["expected_md5"] != NA else NA
                )
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                row.update(
                    status="changed_during_scan",
                    readable=NA,
                    md5_matches=NA,
                    inspection_error="File changed during inspection; rerun after download finishes",
                )
            elif row["md5_matches"] is False:
                row["status"] = "checksum_mismatch"
            elif row["readable"] is False:
                row["status"] = "unreadable"
            else:
                row["status"] = "inspected" if row["expected"] else "unexpected_file"
        except Exception as exc:
            # An I/O or hashing failure is not proof that OpenSlide failed.
            row.update(status="file_io_error", inspection_error=f"{type(exc).__name__}: {exc}")
            log.error("%s - cannot open slide: %s", row["relative_path"], exc)


def annotate_duplicates(rows):
    issues = defaultdict(set)
    checks = ["slide_id", "file_name", "relative_path", "file_id", "expected_md5", "computed_md5"]
    duplicate_report = []
    for key in checks:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            value = row.get(key)
            if value not in (NA, None, ""):
                groups[str(value).casefold()].append(index)
        for value, indices in sorted(groups.items()):
            if len(indices) < 2:
                continue
            for index in indices:
                issues[index].add(key)
            duplicate_report.append(
                {
                    "criterion": key,
                    "value": value,
                    "count": len(indices),
                    "relative_paths": ";".join(rows[i]["relative_path"] for i in indices),
                }
            )
    for i, row in enumerate(rows):
        row["duplicate"] = bool(issues[i])
        row["duplicate_reasons"] = ";".join(sorted(issues[i])) or "none"
    return duplicate_report


def patient_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["expected"] and row["patient_id"] != NA:
            groups[row["patient_id"]].append(row)
    result = []
    for patient_id, items in sorted(groups.items()):
        labels = {r["label"] for r in items if r["label"] in {"Normal", "Tumor"}}
        mixed = len(labels) > 1
        for row in rows:
            if row["patient_id"] == patient_id:
                row["patient_mixed_labels"] = mixed
        result.append(
            {
                "patient_id": patient_id,
                "slide_count": len(items),
                "normal_slides": sum(r["label"] == "Normal" for r in items),
                "tumor_slides": sum(r["label"] == "Tumor" for r in items),
                "mixed_labels": mixed,
                "readable_slides": sum(r["readable"] is True for r in items),
                "readability_not_tested": sum(r["readable"] == NA for r in items),
            }
        )
    return result


def statistics_for(values):
    v = [float(x) for x in values if x != NA and x is not None]
    return (
        {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}
        if v
        else {"n": 0, "min": NA, "median": NA, "max": NA}
    )


def summarize(rows, patients, partials, skipped, mode, verify_md5):
    expected = [r for r in rows if r["expected"]]
    inspected = [r for r in expected if r["metadata_source"] in {"OpenSlide", "recorded OpenSlide"}]
    present = [r for r in expected if r["present"] is True]
    class_counts = dict(Counter(r["label"] for r in expected))
    sizes = {
        c: sum(r["expected_file_size_bytes"] for r in expected if r["label"] == c)
        for c in ["Normal", "Tumor"]
    }
    totals = sum(r["expected_file_size_bytes"] for r in expected)
    missing = sum(r["present"] is False for r in expected)
    invalid = sum(r["label_valid"] is not True for r in rows)
    unknown_patient = sum(r["patient_id"] == NA for r in expected)
    duplicates = sum(r["duplicate"] for r in rows)
    local_failures = sum(
        r["status"]
        in {
            "size_mismatch",
            "checksum_mismatch",
            "unreadable",
            "file_io_error",
            "changed_during_scan",
        }
        for r in rows
    )
    opened = [r for r in inspected if r["metadata_opened"] is True]
    missing_fields = {
        f: sum(r[f] == NA for r in opened)
        for f in ["width", "height", "level_count", "mpp_x", "mpp_y", "objective_power"]
    }
    extras = sum(not r["expected"] for r in rows)
    if mode == "metadata":
        readiness = "NOT_VERIFIED_LOCAL_FILES"
    elif mode == "recorded":
        readiness = "RECORDED_PARTIAL_EVIDENCE_ONLY"
    elif mode == "evidence":
        readiness = "PARTIAL_LOCAL_VERIFICATION"
    elif skipped:
        readiness = "INCOMPLETE_DIRECTORY_SCAN"
    elif missing or partials:
        readiness = "INCOMPLETE_LOCAL_DATASET"
    elif local_failures or invalid or unknown_patient or duplicates or extras or skipped:
        readiness = "REVIEW_REQUIRED"
    elif any(missing_fields.values()):
        readiness = "MISSING_WSI_METADATA_REVIEW"
    else:
        readiness = (
            "PASSED_SAMPLED_READABILITY_AND_MD5"
            if verify_md5
            else "PASSED_SAMPLED_READABILITY_SIZE_ONLY"
        )
    return {
        "audit_mode": mode,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_id": "TCGA-LUSC",
        "expected_slide_count": len(expected),
        "inventory_row_count": len(rows),
        "file_extension_distribution": dict(Counter(r["file_extension"] for r in expected)),
        "missing_patient_id_count": unknown_patient,
        "class_counts": class_counts,
        "unique_patient_count": len(patients),
        "patients_per_class": {
            c: len({r["patient_id"] for r in expected if r["label"] == c and r["patient_id"] != NA})
            for c in ["Normal", "Tumor"]
        },
        "mixed_label_patient_count": sum(p["mixed_labels"] for p in patients),
        "patient_slide_count_distribution": dict(
            sorted(Counter(p["slide_count"] for p in patients).items())
        ),
        "expected_bytes_by_class": sizes,
        "expected_total_bytes": totals,
        "expected_total_gib": totals / 1024**3,
        "expected_total_gb": totals / 10**9,
        "expected_size_mb_statistics": statistics_for(
            r["expected_file_size_bytes"] / 10**6 for r in expected
        ),
        "local_present_expected": len(present) if mode == "local" else NA,
        "expected_slides_with_local_evidence": len(inspected),
        "current_run_inspected_count": sum(
            r["observation_source"] == "current_run" for r in expected
        ),
        "recorded_inspected_count": sum(
            r["observation_source"] == "imported_pilot" for r in expected
        ),
        "local_presence_unknown_count": sum(r["present"] == NA for r in expected),
        "local_missing_expected": missing if mode == "local" and not skipped else NA,
        "unexpected_local_files": extras if mode == "local" else NA,
        "partial_file_count": len(partials) if mode == "local" else NA,
        "directory_scan_errors_or_skipped_links": skipped,
        "local_observed_bytes": sum(
            r["file_size_bytes"] for r in present if r["file_size_bytes"] != NA
        )
        if mode == "local"
        else NA,
        "readable_count": sum(r["readable"] is True for r in expected),
        "unreadable_count": sum(r["readable"] is False for r in expected),
        "readability_not_tested_count": sum(r["readable"] == NA for r in expected),
        "local_problem_count": local_failures,
        "valid_labels": invalid == 0,
        "invalid_label_count": invalid,
        "duplicate_flagged_rows": duplicates,
        "duplicate_scope": "logical identity / declared source MD5; actual file MD5 additionally if requested",
        "verify_md5_requested": verify_md5,
        "md5_pass_count": sum(r["md5_matches"] is True for r in expected),
        "md5_fail_count": sum(r["md5_matches"] is False for r in expected),
        "metadata_inspected_count": len(inspected),
        "metadata_not_inspected_count": len(expected) - len(inspected),
        "metadata_opened_count": len(opened),
        "missing_metadata_denominator": "Only slides whose headers actually opened; excludes untested/failed-open files",
        "missing_fields_among_inspected": missing_fields,
        "objective_power_distribution": dict(Counter(str(r["objective_power"]) for r in inspected)),
        "level_count_distribution": dict(Counter(str(r["level_count"]) for r in inspected)),
        "wsi_numeric_statistics": {
            f: statistics_for(r[f] for r in inspected)
            for f in ["width", "height", "mpp_x", "mpp_y"]
        },
        "readiness": readiness,
        "interpretation": {
            "NA": "Unmeasured, absent or untested: never interpreted as False or zero",
            "readable": "OpenSlide opens file and decodes 64x64 corner/centre regions at every level; not exhaustive pixel QC",
            "normal": "Solid Tissue Normal from a cancer cohort; not a healthy-donor designation",
            "mixed_patient_labels": "Normal and tumor tissue from one patient can coexist; warning for grouped splits, not automatic mislabelling",
            "split_rule": "Keep every slide of each patient_id in one future train/validation/test partition",
            "magnification": "Native WSI metadata; planned 5x processing is not the acquired objective power",
            "duplicate_policy": "Report all matches; never delete or relabel automatically",
            "observation_scope": "In recorded mode, readable/MD5/geometry describe prior evidence, not newly inspected files; current presence is unknown",
        },
    }


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def apply_recorded_evidence(rows, evidence_path, source_hashes):
    """Replay successful pilot observations without claiming a new measurement.

    Only the versioned archive format is accepted. Source hashes and identities
    must match; this validates internal consistency, not the absent image bytes.
    """
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    provenance = evidence["provenance"]
    if evidence["schema_version"] != 1 or provenance["input_sha256"] != source_hashes:
        raise ValueError("Recorded evidence does not match the selected source files")
    timestamp = provenance["recorded_at_utc"]
    if datetime.fromisoformat(timestamp).tzinfo is None:
        raise ValueError("Recorded timestamp must include a timezone")
    by_id = {r["file_id"]: r for r in rows}
    seen = set()
    numeric_fields = [
        "width",
        "height",
        "level_count",
        "mpp_x",
        "mpp_y",
        "objective_power",
        "probe_regions_read",
    ]
    for obs in evidence["observations"]:
        file_id = obs["file_id"]
        if file_id not in by_id or file_id in seen:
            raise ValueError("Unknown or duplicate recorded file ID")
        seen.add(file_id)
        row = by_id[file_id]
        for key in [
            "slide_id",
            "patient_id",
            "label",
            "file_name",
            "expected_md5",
            "expected_file_size_bytes",
        ]:
            if str(row[key]) != obs[key]:
                raise ValueError(f"Recorded identity mismatch: {file_id}: {key}")
        if (
            any(
                obs[k] != "True"
                for k in ["readable", "metadata_opened", "md5_matches", "size_matches"]
            )
            or obs["status"] != "inspected"
            or obs["computed_md5"] != row["expected_md5"]
            or int(obs["file_size_bytes"]) != row["expected_file_size_bytes"]
        ):
            raise ValueError(f"Expected a successful size/MD5/probe observation: {file_id}")
        for key in numeric_fields:
            value = positive_number(obs[key])
            if value == NA:
                raise ValueError(f"Invalid recorded value: {key}")
            if key in {"width", "height", "level_count", "probe_regions_read"}:
                if not value.is_integer():
                    raise ValueError(f"Non-integer recorded value: {key}")
                value = int(value)
            row[key] = value
        dimensions = json.loads(obs["level_dimensions"])
        downsamples = json.loads(obs["level_downsamples"])
        if (
            len(dimensions) != row["level_count"]
            or len(downsamples) != row["level_count"]
            or dimensions[0] != [row["width"], row["height"]]
            or any(len(d) != 2 or any(positive_number(v) == NA for v in d) for d in dimensions)
            or any(positive_number(v) == NA for v in downsamples)
        ):
            raise ValueError("Invalid recorded pyramid geometry")
        row.update(
            readable=True,
            present=NA,
            status="recorded_inspected",
            metadata_opened=True,
            computed_md5=obs["computed_md5"],
            md5_matches=True,
            size_matches=True,
            file_size_bytes=int(obs["file_size_bytes"]),
            file_size_mb=int(obs["file_size_bytes"]) / 10**6,
            file_size_source="recorded filesystem measurement",
            metadata_source="recorded OpenSlide",
            missing_wsi_metadata="none",
            observation_source="imported_pilot",
            observation_time_utc=timestamp,
            level_dimensions=obs["level_dimensions"],
            level_downsamples=obs["level_downsamples"],
        )
    return provenance


def run_audit(
    labels,
    manifest,
    data_root,
    output_dir,
    mode="local",
    verify_md5=False,
    opener=None,
    snapshot=None,
    recorded_evidence=None,
):
    if mode not in {"local", "metadata", "evidence", "recorded"}:
        raise ValueError(f"Unknown audit mode: {mode}")
    if mode in {"metadata", "recorded"} and (data_root is not None or verify_md5):
        raise ValueError("Offline modes do not use --data-root or --verify-md5")
    if recorded_evidence is not None and mode != "recorded":
        raise ValueError("--recorded-evidence requires --mode recorded")
    output_dir = Path(output_dir)
    data_root = Path(data_root) if data_root is not None else None
    expected = load_expected(labels, manifest, snapshot)
    source_hashes = {"labels": hash_file(labels), "manifest": hash_file(manifest)}
    if snapshot:
        source_hashes["gdc_snapshot"] = hash_file(snapshot)
    if mode in {"local", "evidence"} and (data_root is None or not data_root.is_dir()):
        raise ValueError(
            f"Data directory not found: {data_root}. Use --data-root with the actual WSI folder."
        )
    if mode in {"local", "evidence"} and opener is None:
        import openslide

        opener = openslide.OpenSlide
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"wsi_audit_{id(expected)}")
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "audit.log", mode="w", encoding="utf-8"),
    ]:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(handler)
    try:
        if mode in {"metadata", "recorded"}:
            rows, partials, skipped = expected, [], []
            for r in rows:
                r.update(
                    file_size_bytes=r["expected_file_size_bytes"],
                    file_size_mb=round(r["expected_file_size_bytes"] / 10**6, 6),
                    file_size_source="GDC manifest, not local measurement",
                )
            if mode == "recorded":
                if recorded_evidence is None:
                    raise ValueError("Recorded mode needs a pilot evidence file")
                provenance = apply_recorded_evidence(rows, recorded_evidence, source_hashes)
        else:
            paths, partials, skipped = scan_paths(data_root, log)
            rows = match_paths(expected, data_root, paths)
            if skipped:
                for row in rows:
                    if row["present"] is False:
                        row.update(present=NA, status="not_located_in_incomplete_scan")
            audit_local_rows(rows, data_root, opener, verify_md5, log)
            if mode == "evidence":
                for row in rows:
                    if row["present"] is False:
                        row.update(present=NA, status="not_inspected")
                    if row["file_size_bytes"] == NA:
                        row.update(
                            file_size_bytes=row["expected_file_size_bytes"],
                            file_size_mb=round(row["expected_file_size_bytes"] / 10**6, 6),
                            file_size_source="GDC manifest, not local measurement",
                        )
        duplicates = annotate_duplicates(rows)
        patients = patient_summary(rows)
        summary = summarize(rows, patients, partials, skipped, mode, verify_md5)
        summary["input_sha256"] = source_hashes
        if mode == "recorded":
            summary["recorded_evidence_provenance"] = provenance
            summary["recorded_evidence_sha256"] = hash_file(recorded_evidence)
        summary["snapshot_crosscheck"] = "passed" if snapshot else "not_requested"
        if snapshot:
            summary["input_sha256"]["gdc_snapshot"] = hash_file(snapshot)
        summary["software"] = {"python": sys.version.split()[0]}
        for package in ["openslide-python", "openslide-bin", "Pillow"]:
            try:
                summary["software"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                summary["software"][package] = "not installed"
        if summary["mixed_label_patient_count"]:
            log.warning(
                "%s patients have both Normal and Tumor slides; preserve patient grouping in all future splits.",
                summary["mixed_label_patient_count"],
            )
        rows.sort(key=lambda r: (r["file_id"], r["relative_path"]))
        write_csv(output_dir / "dataset_inventory.csv", rows, INVENTORY_FIELDS)
        write_csv(
            output_dir / "patient_summary.csv",
            patients,
            [
                "patient_id",
                "slide_count",
                "normal_slides",
                "tumor_slides",
                "mixed_labels",
                "readable_slides",
                "readability_not_tested",
            ],
        )
        write_csv(
            output_dir / "mixed_label_patients.csv",
            [p for p in patients if p["mixed_labels"]],
            ["patient_id", "slide_count", "normal_slides", "tumor_slides", "mixed_labels"],
        )
        write_csv(
            output_dir / "issues.csv",
            [
                r
                for r in rows
                if r["duplicate"]
                or r["label_valid"] is not True
                or r["patient_id"] == NA
                or r["status"] not in {"inspected", "not_inspected", "recorded_inspected"}
                or r["missing_wsi_metadata"] not in {NA, "none"}
            ],
            INVENTORY_FIELDS,
        )
        write_csv(
            output_dir / "duplicates.csv",
            duplicates,
            ["criterion", "value", "count", "relative_paths"],
        )
        write_csv(
            output_dir / "partial_files.csv",
            [{"relative_path": p} for p in partials],
            ["relative_path"],
        )
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log.info(
            "Audit complete: %s expected slides; %s patients; status=%s",
            len(expected),
            len(patients),
            summary["readiness"],
        )
        log.info(
            "Classes: %s; readable=%s; unreadable=%s; untested=%s; duplicates=%s",
            summary["class_counts"],
            summary["readable_count"],
            summary["unreadable_count"],
            summary["readability_not_tested_count"],
            summary["duplicate_flagged_rows"],
        )
        return summary
    finally:
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, help="Actual dataset root, required for local/evidence modes"
    )
    parser.add_argument("--labels", type=Path, default=SOURCE / "labels_TCGA_LUSC_694.csv")
    parser.add_argument("--manifest", type=Path, default=SOURCE / "gdc_manifest_TCGA_LUSC_694.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "dataset-audit")
    parser.add_argument(
        "--mode",
        choices=["local", "metadata", "evidence", "recorded"],
        default="recorded",
        help="recorded: replay historical pilot; metadata: no images; evidence: scan a subset; local: complete root",
    )
    parser.add_argument(
        "--recorded-evidence",
        type=Path,
        help="Versioned pilot observations JSON (recorded mode only)",
    )
    parser.add_argument("--snapshot", type=Path, default=SOURCE / "gdc_metadata_snapshot.json")
    parser.add_argument(
        "--verify-md5",
        action="store_true",
        help="Read complete files to verify GDC MD5 and detect identical content",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit 3 if local readiness checks do not pass"
    )
    args = parser.parse_args(argv)
    try:
        evidence = args.recorded_evidence
        if args.mode == "recorded" and evidence is None:
            evidence = SOURCE / "pilot_observations.json"
        summary = run_audit(
            args.labels,
            args.manifest,
            args.data_root,
            args.output_dir,
            args.mode,
            args.verify_md5,
            snapshot=args.snapshot,
            recorded_evidence=evidence,
        )
    except (ValueError, OSError, ImportError, csv.Error, KeyError, TypeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.strict and not summary["readiness"].startswith("PASSED_"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
