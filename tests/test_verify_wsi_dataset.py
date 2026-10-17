import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py

from scripts import run_wsi_dataset as runner
from scripts import verify_wsi_dataset as verifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_verify_pilot14 import SlideFixture, _FakeSlide, _write_csv


def _controls(root, count=5):
    identities = [
        {
            "slide_id": f"slide-{index}", "patient_id": "shared-patient",
            "label": "Normal", "resolved_path": str(root / f"{index}.svs"),
            "source_size_bytes": 10, "source_mtime_ns": index,
            "expected_md5": "a" * 32, "computed_md5": "a" * 32,
        }
        for index in range(count)
    ]
    settings = {"preprocessing": {"min_tissue_percentage": 20}, "seed": 42}
    config = {
        **settings, "base_settings": settings,
        "selected_slide_ids": [row["slide_id"] for row in identities],
        "source_identities": identities,
        "slide_fingerprints": {
            row["slide_id"]: runner._slide_fingerprint(settings, row) for row in identities
        },
    }
    config["run_fingerprint"] = runner.pilot._run_fingerprint({
        "settings": settings, "selected_slide_ids": config["selected_slide_ids"],
    })
    manifest = [
        {
            "slide_id": row["slide_id"], "patient_id": row["patient_id"],
            "label": row["label"], "slide_path": row["resolved_path"],
            "source_size_bytes": row["source_size_bytes"],
            "source_md5": row["computed_md5"], "status": "success",
            "run_fingerprint": config["slide_fingerprints"][row["slide_id"]],
        }
        for row in identities
    ]
    return copy.deepcopy(identities), manifest, config


class DatasetVerificationTests(unittest.TestCase):
    def test_arbitrary_unbalanced_cohort_and_repeated_patients_are_allowed(self):
        selection, manifest, config = _controls(Path("cohort"))
        self.assertEqual(verifier._validate_control_files(selection, manifest, config), [])
        self.assertEqual(verifier._validate_fingerprints(config), [])

    def test_missing_manifest_slide_and_extra_source_identity_fail_coverage(self):
        selection, manifest, config = _controls(Path("cohort"))
        errors = verifier._validate_control_files(selection, manifest[:-1], config)
        self.assertIn("Selection and manifest slide IDs/order differ", errors)
        extra = dict(config["source_identities"][0], slide_id="extra-slide")
        config["source_identities"].append(extra)
        errors = verifier._validate_control_files(selection, manifest, config)
        self.assertIn("run_config source identities do not match selected slide IDs", errors)

    def test_identity_and_per_slide_fingerprint_tampering_are_rejected(self):
        selection, manifest, config = _controls(Path("cohort"))
        manifest[0]["run_fingerprint"] = "wrong"
        selection[1]["patient_id"] = "wrong-patient"
        errors = verifier._validate_control_files(selection, manifest, config)
        self.assertTrue(any("Manifest slide fingerprint mismatch" in error for error in errors))
        self.assertTrue(any("selection slide-1 mismatch for patient_id" in error for error in errors))
        config["source_identities"][0]["computed_md5"] = "b" * 32
        self.assertTrue(verifier._validate_fingerprints(config))

    def test_top_level_scientific_settings_must_match_fingerprinted_settings(self):
        _, _, config = _controls(Path("cohort"))
        config["preprocessing"] = {"min_tissue_percentage": 5}
        errors = verifier._validate_fingerprints(config)
        self.assertTrue(any("preprocessing differs" in error for error in errors))

    def test_independent_dataset_rescan_rejects_an_unselected_wsi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selected.svs").write_bytes(b"selected")
            (root / "missed.svs").write_bytes(b"missed")
            errors = verifier._validate_dataset_coverage(
                [{"resolved_path": str(root / "selected.svs")}],
                {"dataset_root": str(root), "local_wsi_files_discovered": 1},
            )
            self.assertTrue(any("coverage differs" in error for error in errors))
            self.assertTrue(any("local_wsi_files_discovered" in error for error in errors))

    def test_per_slide_adapter_verifies_real_payload_and_catches_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            config = dict(fixture.run_config, run_fingerprint="cohort-id",
                          slide_fingerprints={fixture.slide_id: "abc123"})
            with mock.patch.object(verifier.scientific, "open_slide", return_value=_FakeSlide()), mock.patch.object(
                verifier.scientific, "extract_patch", return_value=fixture.original_image,
            ):
                detail, failure = verifier._verify_one(fixture.row, fixture.root, config, True, 3)
            self.assertIsNone(failure)
            self.assertEqual(detail["patch_count"], 1)
            with h5py.File(fixture.hdf5_path, "r+") as file:
                file["images"][0, 0, 0, 0, 0] = 123
            detail, failure = verifier._verify_one(fixture.row, fixture.root, config, True, 3)
            self.assertIsNone(detail)
            self.assertIn("HDF5 SHA256 mismatch", failure["error_message"])

    def test_missing_control_files_still_emit_failed_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            report = verifier.verify(directory)
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(len(report["errors"]), 3)
            saved = json.loads((Path(directory) / "verification.json").read_text())
            self.assertEqual(saved["status"], "FAILED")
            self.assertTrue((Path(directory) / "verification.md").is_file())

    def test_end_to_end_report_with_all_scientific_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            settings = {
                "preprocessing": fixture.preprocessing_config,
                "save_qc": False, "qc_sample_count": 1,
                "pipeline_code_sha256": runner._dataset_code_fingerprint(),
            }
            fingerprint = runner._slide_fingerprint(settings, fixture.identity)
            config = {
                **settings, "base_settings": settings,
                "source_identities": [fixture.identity],
                "selected_slide_ids": [fixture.slide_id],
                "slide_fingerprints": {fixture.slide_id: fingerprint},
                "dataset_root": str(fixture.root), "local_wsi_files_discovered": 1,
            }
            config["run_fingerprint"] = runner.pilot._run_fingerprint({
                "settings": settings, "selected_slide_ids": config["selected_slide_ids"],
            })
            with h5py.File(fixture.hdf5_path, "r+") as file:
                file.attrs["run_fingerprint"] = fingerprint
            fixture.row["run_fingerprint"] = fingerprint
            fixture.row["augmentation_size_bytes"] = str(fixture.hdf5_path.stat().st_size)
            fixture.row["augmentation_sha256"] = verifier.scientific._hash_file(fixture.hdf5_path)
            (fixture.slide_dir / "complete.json").write_text(json.dumps({
                "status": "success", "run_fingerprint": fingerprint,
                "source_identity": fixture.identity, "manifest_row": fixture.row,
            }), encoding="utf-8")
            (fixture.root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            _write_csv(fixture.root / "selection.csv", list(fixture.identity), [fixture.identity])
            _write_csv(fixture.root / "manifest.csv", list(fixture.row), [fixture.row])
            with mock.patch.object(verifier.scientific, "open_slide", return_value=_FakeSlide()), mock.patch.object(
                verifier.scientific, "extract_patch", return_value=fixture.original_image,
            ):
                report = verifier.verify(directory, verify_source_md5=True, source_sample_count=100000)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["status"], "PASSED")
            self.assertEqual(report["slides"], 1)
            self.assertEqual(report["slide_details"][0]["source_reextract_samples"], 1)


if __name__ == "__main__":
    unittest.main()
