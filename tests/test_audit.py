"""Regression tests for audit failure modes; no real WSI or network required."""

import csv
import errno
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lung_wsi_mil.audit.inventory import (
    NA,
    SOURCE,
    apply_recorded_evidence,
    hash_file,
    inspect_slide,
    load_expected,
    main,
    positive_number,
    read_csv,
    run_audit,
)
from lung_wsi_mil.audit.sources import validate


class Image:
    def load(self):
        pass

    def close(self):
        pass


class Slide:
    dimensions = (1000, 800)
    level_count = 2
    level_dimensions = ((1000, 800), (250, 200))
    level_downsamples = (1.0, 4.0)
    properties = {
        "openslide.mpp-x": "0.5",
        "openslide.mpp-y": "0.5",
        "openslide.objective-power": "20",
    }

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.read_bytes().startswith(b"bad!"):
            raise OSError("corrupt WSI")
        return self

    def __exit__(self, *args):
        pass

    def read_region(self, location, level, size):
        return Image()


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "data"
        self.root.mkdir()
        self.out = self.base / "out"
        self.labels = self.base / "labels.csv"
        self.manifest = self.base / "manifest.txt"
        self.entries = []
        for i, (sample, label) in enumerate([("11A", "0"), ("01A", "1")], 1):
            uuid = f"00000000-0000-0000-0000-{i:012d}"
            name = f"TCGA-AA-0001-{sample}-01-TS1.uuid.svs"
            payload = (f"good{i}").encode()
            folder = self.root / uuid
            folder.mkdir()
            (folder / name).write_bytes(payload)
            self.entries.append(
                {
                    "file_id": uuid,
                    "file_name": name,
                    "label": label,
                    "class": "normal" if label == "0" else "tumor",
                    "case_submitter_id": "TCGA-AA-0001",
                    "file_size_bytes": str(len(payload)),
                    "md5sum": hashlib.md5(payload).hexdigest(),
                }
            )
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self):
        with self.labels.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.entries[0]))
            w.writeheader()
            w.writerows(self.entries)
        with self.manifest.open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["id", "filename", "md5", "size", "state"], delimiter="\t"
            )
            w.writeheader()
            for e in self.entries:
                w.writerow(
                    dict(
                        id=e["file_id"],
                        filename=e["file_name"],
                        md5=e["md5sum"],
                        size=e["file_size_bytes"],
                        state="released",
                    )
                )

    def path(self, index=0):
        e = self.entries[index]
        return self.root / e["file_id"] / e["file_name"]

    def run_case(self, **kwargs):
        root = None if kwargs.get("mode") in {"metadata", "recorded"} else self.root
        return run_audit(self.labels, self.manifest, root, self.out, opener=Slide, **kwargs)

    def test_complete_and_patient_overlap(self):
        s = self.run_case(verify_md5=True)
        self.assertEqual(s["readiness"], "PASSED_SAMPLED_READABILITY_AND_MD5")
        self.assertEqual(s["readable_count"], 2)
        self.assertEqual(s["mixed_label_patient_count"], 1)
        self.assertEqual(s["unique_patient_count"], 1)
        self.assertEqual(
            read_csv(self.out / "dataset_inventory.csv")[0]["probe_regions_read"], "10"
        )

    def test_corrupt_file_does_not_stop_next(self):
        self.path().write_bytes(b"bad!!")
        s = self.run_case()
        self.assertEqual(s["unreadable_count"], 1)
        self.assertEqual(s["readable_count"], 1)
        self.assertEqual(s["metadata_opened_count"], 1)

    def test_truncated_file_is_not_success(self):
        self.path().write_bytes(b"x")
        s = self.run_case()
        self.assertEqual(s["local_problem_count"], 1)
        self.assertEqual(s["readability_not_tested_count"], 1)
        self.assertEqual(s["readiness"], "REVIEW_REQUIRED")

    def test_wrong_md5_detected(self):
        self.path().write_bytes(b"wrong")
        s = self.run_case(verify_md5=True)
        self.assertEqual(s["md5_fail_count"], 1)
        self.assertEqual(s["readiness"], "REVIEW_REQUIRED")

    def test_missing_slide_and_partial_are_separate(self):
        self.path().unlink()
        (self.root / "download.partial").write_bytes(b"abc")
        s = self.run_case()
        self.assertEqual(s["local_missing_expected"], 1)
        self.assertEqual(s["partial_file_count"], 1)
        self.assertEqual(s["unreadable_count"], 0)

    def test_evidence_mode_does_not_claim_missing_on_company_pc(self):
        self.path().unlink()
        s = self.run_case(mode="evidence")
        self.assertEqual(s["local_missing_expected"], NA)
        self.assertEqual(s["readability_not_tested_count"], 1)
        self.assertEqual(s["readiness"], "PARTIAL_LOCAL_VERIFICATION")
        self.assertEqual(read_csv(self.out / "dataset_inventory.csv")[0]["present"], NA)

    def test_duplicate_copy_retained_and_flagged(self):
        extra = self.root / "copy"
        extra.mkdir()
        shutil.copyfile(self.path(), extra / self.path().name)
        s = self.run_case(verify_md5=True)
        self.assertEqual(s["inventory_row_count"], 3)
        self.assertEqual(s["duplicate_flagged_rows"], 2)
        self.assertTrue((extra / self.path().name).exists())
        self.assertEqual(s["unique_patient_count"], 1)

    def test_unknown_slide_has_no_invented_label(self):
        (self.root / "unknown.TIFF").write_bytes(b"goodX")
        s = self.run_case()
        self.assertEqual(s["unexpected_local_files"], 1)
        row = [
            r
            for r in read_csv(self.out / "dataset_inventory.csv")
            if r["file_name"] == "unknown.TIFF"
        ][0]
        self.assertEqual(row["label"], NA)
        self.assertEqual(row["patient_id"], NA)

    def test_conflicting_labels_are_flagged(self):
        self.entries[0]["label"] = "1"
        self.write_inputs()
        s = self.run_case()
        self.assertEqual(s["invalid_label_count"], 1)
        self.assertEqual(read_csv(self.out / "dataset_inventory.csv")[0]["label"], NA)

    def test_missing_optional_metadata_is_na(self):
        class Missing(Slide):
            properties = {}

        result = inspect_slide(self.path(), Missing)
        self.assertTrue(result["readable"])
        self.assertEqual(result["mpp_x"], NA)
        self.assertEqual(result["objective_power"], NA)

    def test_invalid_numbers_never_guessed(self):
        for value in [None, "NaN", "Inf", "-2", "0", "invalid"]:
            self.assertEqual(positive_number(value), NA)

    def test_missing_patient_blocks_ready(self):
        old = self.path()
        self.entries[0]["file_name"] = "unknown.svs"
        self.entries[0]["case_submitter_id"] = ""
        old.rename(old.parent / "unknown.svs")
        self.write_inputs()
        s = self.run_case()
        self.assertEqual(s["missing_patient_id_count"], 1)
        self.assertEqual(s["readiness"], "REVIEW_REQUIRED")

    def test_repeated_run_identical_inventory(self):
        self.run_case()
        first = read_csv(self.out / "dataset_inventory.csv")
        self.run_case()
        second = read_csv(self.out / "dataset_inventory.csv")
        for rows in [first, second]:
            for row in rows:
                row.pop("observation_time_utc")
        self.assertEqual(first, second)

    def test_metadata_mode_does_not_mark_unreadable(self):
        s = self.run_case(mode="metadata")
        self.assertEqual(s["readable_count"], 0)
        self.assertEqual(s["unreadable_count"], 0)
        self.assertEqual(s["readability_not_tested_count"], 2)

    def test_ambiguous_source_ids_fail_early(self):
        self.entries[1]["file_id"] = self.entries[0]["file_id"]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate file IDs"):
            load_expected(self.labels, self.manifest)

    def test_skipped_directory_never_declares_missing(self):
        self.path().unlink()
        try:
            (self.root / "linked").symlink_to(self.base, target_is_directory=True)
        except OSError as exc:
            if exc.errno == errno.EPERM or getattr(exc, "winerror", None) == 1314:
                self.skipTest(f"Directory symlink privilege is unavailable: {exc}")
            raise
        summary = self.run_case()
        self.assertEqual(summary["readiness"], "INCOMPLETE_DIRECTORY_SCAN")
        self.assertEqual(summary["local_missing_expected"], NA)
        self.assertEqual(summary["local_presence_unknown_count"], 1)
        row = read_csv(self.out / "dataset_inventory.csv")[0]
        self.assertEqual(row["status"], "not_located_in_incomplete_scan")
        self.assertEqual(row["readable"], NA)

    def test_walk_error_blocks_completeness(self):
        def broken_walk(root, **kwargs):
            kwargs["onerror"](PermissionError(13, "Permission denied", str(root / "locked")))
            return iter([])

        with patch("lung_wsi_mil.audit.inventory.os.walk", side_effect=broken_walk):
            summary = self.run_case()
        self.assertEqual(summary["local_missing_expected"], NA)
        self.assertEqual(summary["readiness"], "INCOMPLETE_DIRECTORY_SCAN")
        self.assertEqual(
            summary["directory_scan_errors_or_skipped_links"], ["locked: Permission denied"]
        )

    def test_hash_failure_does_not_reclassify_readable_as_corrupt(self):
        def fail_md5(path, algorithm="sha256"):
            if algorithm == "md5":
                raise OSError("read denied during hashing")
            return hash_file(path, algorithm)

        with patch("lung_wsi_mil.audit.inventory.hash_file", side_effect=fail_md5):
            summary = self.run_case(verify_md5=True)
        self.assertEqual(summary["readable_count"], 2)
        self.assertEqual(summary["unreadable_count"], 0)
        self.assertEqual(summary["local_problem_count"], 2)

    def test_late_region_decode_failure_and_level0_coordinates(self):
        calls = []

        class LateFailure(Slide):
            def read_region(self, location, level, size):
                calls.append((location, level, size))
                if level == 1 and location == (744, 544):
                    raise OSError("bad pyramid tile")
                return Image()

        result = inspect_slide(self.path(), LateFailure)
        self.assertFalse(result["readable"])
        self.assertTrue(result["metadata_opened"])
        self.assertEqual(result["probe_regions_read"], 9)
        self.assertIn(((744, 544), 1, (64, 64)), calls)

    def test_failed_open_metadata_is_unobserved(self):
        self.path().write_bytes(b"bad!!")
        result = inspect_slide(self.path(), Slide)
        self.assertEqual(result["missing_wsi_metadata"], NA)

    def test_bad_cli_arguments_and_strict_exit(self):
        self.assertEqual(main(["--mode", "local", "--data-root", str(self.base / "absent")]), 2)
        self.assertEqual(main(["--mode", "metadata", "--verify-md5"]), 2)
        self.assertEqual(main(["--output-dir", str(self.out), "--strict"]), 3)


class FrozenCohortTests(unittest.TestCase):
    def test_snapshot_selection_and_patient_grouping(self):
        self.assertEqual(validate()["status"], "PASSED")
        with tempfile.TemporaryDirectory() as d:
            summary = run_audit(
                SOURCE / "labels_TCGA_LUSC_694.csv",
                SOURCE / "gdc_manifest_TCGA_LUSC_694.txt",
                None,
                d,
                mode="recorded",
                snapshot=SOURCE / "gdc_metadata_snapshot.json",
                recorded_evidence=SOURCE / "pilot_observations.json",
            )
            self.assertEqual(summary["class_counts"], {"Normal": 347, "Tumor": 347})
            self.assertEqual(summary["unique_patient_count"], 373)
            self.assertEqual(summary["mixed_label_patient_count"], 153)
            self.assertEqual(summary["readable_count"], 2)
            self.assertEqual(summary["recorded_inspected_count"], 2)
            self.assertEqual(summary["current_run_inspected_count"], 0)
            self.assertEqual(summary["readability_not_tested_count"], 692)
            self.assertEqual(summary["local_presence_unknown_count"], 694)
            self.assertEqual(summary["readiness"], "RECORDED_PARTIAL_EVIDENCE_ONLY")
            self.assertEqual(
                (Path(d) / "dataset_inventory.csv").read_bytes(),
                (SOURCE.parent / "dataset_inventory.csv").read_bytes(),
            )

    def test_recorded_evidence_rejects_tampering(self):
        rows = load_expected(
            SOURCE / "labels_TCGA_LUSC_694.csv",
            SOURCE / "gdc_manifest_TCGA_LUSC_694.txt",
            SOURCE / "gdc_metadata_snapshot.json",
        )
        original = json.loads((SOURCE / "pilot_observations.json").read_text())
        mutations = [
            lambda x: x["provenance"]["input_sha256"].update(labels="wrong"),
            lambda x: x["observations"][0].update(patient_id="wrong"),
            lambda x: x["observations"].append(x["observations"][0]),
            lambda x: x["observations"][0].update(computed_md5="wrong"),
        ]
        for mutate in mutations:
            changed = json.loads(json.dumps(original))
            mutate(changed)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "evidence.json"
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    apply_recorded_evidence(rows, p, original["provenance"]["input_sha256"])

    def test_snapshot_csv_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            content = (SOURCE / "labels_TCGA_LUSC_694.csv").read_text(encoding="utf-8-sig")
            p.write_text(content)
            rows = read_csv(p)
            rows[0]["case_id"] = "wrong"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            with self.assertRaisesRegex(ValueError, "CSV/snapshot mismatch"):
                load_expected(
                    p,
                    SOURCE / "gdc_manifest_TCGA_LUSC_694.txt",
                    SOURCE / "gdc_metadata_snapshot.json",
                )


if __name__ == "__main__":
    unittest.main()
