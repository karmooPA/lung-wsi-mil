import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lung_wsi_mil.workflows import dataset as dataset
from lung_wsi_mil.workflows import pilot as pilot
from tests import test_pilot14 as pilot_fixtures


class DatasetSelectionTests(unittest.TestCase):
    def _rows(self, root, count=3):
        rows = []
        for index in range(count):
            path = root / f"slide-{index}.svs"
            path.write_bytes(f"slide-{index}".encode("ascii"))
            rows.append(
                {
                    "slide_id": f"slide-{index}",
                    "patient_id": "same-patient",
                    "label": "Normal" if index % 2 else "Tumor",
                    "relative_path": path.name,
                    "duplicate": "false",
                    "label_valid": "true",
                    "expected_file_size_bytes": str(path.stat().st_size),
                    "expected_md5": pilot._hash_file(path, "md5"),
                }
            )
        return rows

    def _select(self, root, rows):
        dataset_root, paths, by_name, by_prefix = pilot._index_local_slides(root)
        return dataset._select_all_present(rows, dataset_root, paths, by_name, by_prefix)

    def test_all_slides_selected_with_repeated_patients_and_unbalanced_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(root)
            selected = self._select(root, rows)
            self.assertEqual(len(selected), 3)
            self.assertEqual({row["patient_id"] for row in selected}, {"same-patient"})
            self.assertEqual(
                {row["slide_id"] for row in selected}, {"slide-0", "slide-1", "slide-2"}
            )

    def test_unknown_physical_slide_is_rejected_instead_of_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(root)
            (root / "unlisted.svs").write_bytes(b"unknown")
            with self.assertRaisesRegex(ValueError, "does not unambiguously cover 1"):
                self._select(root, rows)

    def test_duplicate_paths_ids_and_directory_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(root)
            with self.assertRaisesRegex(ValueError, "Duplicate present slide_id"):
                self._select(root, [*rows, rows[0]])
            shared = dict(rows[0], slide_id="different-id")
            with self.assertRaisesRegex(ValueError, "Duplicate physical WSI"):
                self._select(root, [*rows, shared])
            collided = [dict(rows[0], slide_id="A:B"), dict(rows[1], slide_id="A?B"), rows[2]]
            with self.assertRaisesRegex(ValueError, "directory name collision"):
                self._select(root, collided)

    def test_invalid_present_label_is_rejected_instead_of_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows(root)
            rows[0]["label"] = "Unknown"
            with self.assertRaisesRegex(ValueError, "unsupported label"):
                self._select(root, rows)

    def test_slide_fingerprint_depends_on_scientific_settings_and_own_source(self):
        settings = {"seed": 42, "preprocessing": {"magnification": 5}}
        identity = {"slide_id": "a", "source_size_bytes": 10}
        fingerprint = dataset._slide_fingerprint(settings, identity)
        # Adding other cohort members does not appear in the slide input.
        self.assertEqual(fingerprint, dataset._slide_fingerprint(dict(settings), dict(identity)))
        self.assertNotEqual(
            fingerprint, dataset._slide_fingerprint(dict(settings, seed=43), identity)
        )
        self.assertNotEqual(
            fingerprint, dataset._slide_fingerprint(settings, dict(identity, source_size_bytes=11))
        )

    def test_disk_guard_accounts_for_patch_bytes_and_concurrent_workers(self):
        required = dataset.GIB + 10 * dataset.BYTES_PER_PATCH * 2
        with mock.patch.object(
            dataset.shutil, "disk_usage", return_value=SimpleNamespace(free=required - 1)
        ):
            with self.assertRaisesRegex(OSError, "Insufficient output disk space"):
                dataset._ensure_disk_space("unused", min_free_gib=1, patch_count=10, workers=2)
        with mock.patch.object(
            dataset.shutil, "disk_usage", return_value=SimpleNamespace(free=required)
        ):
            self.assertEqual(
                dataset._ensure_disk_space("unused", min_free_gib=1, patch_count=10, workers=2),
                required,
            )

    def test_source_md5_failure_is_recorded_without_opening_wsi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            outputs = root / "outputs"
            outputs.mkdir()
            row = self._select(inputs, self._rows(inputs, count=1))[0]
            row["expected_md5"] = "0" * 32
            options = {"min_free_gib": 0, "verify_source_md5": True}
            with mock.patch.object(pilot, "preprocess_slide") as process:
                result = dataset._process_slide(row, outputs, options, {}, {})
            self.assertEqual(result["manifest_row"]["status"], "failed")
            self.assertIn("MD5 mismatch", result["manifest_row"]["error_message"])
            self.assertTrue((outputs / "failures" / "slide-0.json").is_file())
            process.assert_not_called()

    def test_processing_writes_valid_artifacts_and_resume_survives_new_cohort_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            outputs = root / "outputs"
            outputs.mkdir()
            row = self._select(inputs, self._rows(inputs, count=1))[0]
            options = vars(
                dataset.parse_args(
                    [
                        str(inputs),
                        str(outputs),
                        "--min-free-gib",
                        "0",
                        "--verify-source-md5",
                        "--no-qc",
                    ]
                )
            )
            config = pilot.build_config(5, 512, 20, 35, 1024, save_qc=False)
            fixture = pilot_fixtures.PilotStorageIntegrationTests()
            settings = {"seed": 42, "preprocessing": config}
            dataset._initialize_worker()
            with (
                mock.patch.object(pilot, "preprocess_slide", return_value=fixture._result()),
                mock.patch.object(pilot, "open_slide", return_value=pilot_fixtures._FakeSlide()),
                mock.patch.object(pilot, "extract_patch", side_effect=fixture._patch),
            ):
                first = dataset._process_slide(row, outputs, options, settings, config)
            self.assertEqual(first["manifest_row"]["status"], "success")
            self.assertTrue((outputs / "slides" / "slide-0" / "complete.json").is_file())
            (inputs / "new-cohort-member.svs").write_bytes(b"another slide")
            old_failure = outputs / "failures" / "slide-0.json"
            old_failure.parent.mkdir()
            old_failure.write_text('{"status":"failed"}')
            with mock.patch.object(pilot, "preprocess_slide") as process:
                resumed = dataset._process_slide(row, outputs, options, settings, config)
            self.assertTrue(resumed["resumed"])
            self.assertEqual(first["run_fingerprint"], resumed["run_fingerprint"])
            self.assertFalse(old_failure.exists())
            process.assert_not_called()

    def test_stale_working_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            outputs = root / "outputs"
            outputs.mkdir()
            row = self._select(inputs, self._rows(inputs, count=1))[0]
            options = vars(
                dataset.parse_args([str(inputs), str(outputs), "--min-free-gib", "0", "--force"])
            )
            import os

            stale = outputs / "slides" / f".slide-0.working.{os.getpid()}"
            stale.mkdir(parents=True)
            (stale / "evidence.txt").write_text("previous interrupted run")
            result = dataset._process_slide(row, outputs, options, {}, {})
            self.assertEqual(result["manifest_row"]["status"], "failed")
            self.assertTrue((stale / "evidence.txt").is_file())


if __name__ == "__main__":
    unittest.main()
