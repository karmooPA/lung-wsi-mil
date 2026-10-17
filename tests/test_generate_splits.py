"""Regression checks for safe, portable split generation."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.generate_splits import METADATA_PATH, PROJECT, generate_splits


class GenerateSplitsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metadata_path = self.root / "cohort.csv"
        self.metadata = pd.DataFrame([
            {"slide_id": f"patient_{patient:02d}_{label}",
             "patient_id": f"patient_{patient:02d}", "label": label}
            for patient in range(40)
            for label in ("Normal", "Tumor")
        ])
        self.metadata.to_csv(self.metadata_path, index=False)

    def test_default_metadata_uses_repository_inventory(self):
        self.assertEqual(METADATA_PATH, PROJECT / "metadata" / "dataset_inventory.csv")
        self.assertTrue(METADATA_PATH.is_file())

    def test_generation_is_complete_and_reproducible(self):
        first = generate_splits(self.metadata_path, self.root / "first")
        second = generate_splits(self.metadata_path, self.root / "second")
        first_files = sorted(path.relative_to(first) for path in first.rglob("*.csv"))
        self.assertEqual(len(first_files), 60)
        self.assertEqual(first_files, sorted(path.relative_to(second) for path in second.rglob("*.csv")))
        for relative in first_files:
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        train = pd.read_csv(first / "seed_2" / "train_cv.csv")
        test = pd.read_csv(first / "seed_2" / "test.csv")
        self.assertEqual(set(train["slide_id"]) | set(test["slide_id"]), set(self.metadata["slide_id"]))
        self.assertTrue(set(train["patient_id"]).isdisjoint(test["patient_id"]))

    def test_existing_output_is_preserved_without_reading_input(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "original.csv"
        sentinel.write_bytes(b"existing results\n")
        with self.assertRaisesRegex(FileExistsError, "Choose a new"):
            generate_splits(self.root / "missing.csv", output)
        self.assertEqual(sentinel.read_bytes(), b"existing results\n")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_frozen_cohort_reproduces_all_stored_split_memberships(self):
        # sklearn <1.8 shuffled group counts without remapping group identities.
        # A seed-only test misses that change; compare the accepted partitions.
        output = generate_splits(METADATA_PATH, self.root / "frozen-cohort")
        expected_files = sorted((PROJECT / "splits").rglob("*.csv"))
        self.assertEqual(len(expected_files), 60)
        for expected_path in expected_files:
            relative = expected_path.relative_to(PROJECT / "splits")
            with self.subTest(split=str(relative)):
                columns = ["slide_id", "patient_id", "label"]
                expected = pd.read_csv(expected_path)[columns].sort_values("slide_id").reset_index(drop=True)
                actual = pd.read_csv(output / relative)[columns].sort_values("slide_id").reset_index(drop=True)
                pd.testing.assert_frame_equal(actual, expected)

    def test_invalid_metadata_does_not_publish_outputs(self):
        invalid = [
            (self.metadata.drop(columns="patient_id"), "Missing required columns"),
            (pd.concat([self.metadata, self.metadata.iloc[:1]]), "duplicate slide_id"),
            (self.metadata.assign(patient_id=" "), "empty patient_id"),
            (self.metadata.assign(label="Tumor"), "exactly two labels"),
        ]
        for metadata, message in invalid:
            with self.subTest(message=message):
                metadata.to_csv(self.metadata_path, index=False)
                output = self.root / "invalid-output"
                with self.assertRaisesRegex(ValueError, message):
                    generate_splits(self.metadata_path, output)
                self.assertFalse(output.exists())

    def test_failed_generation_cleans_staging_and_publishes_nothing(self):
        with patch("scripts.generate_splits.save_all_splits", side_effect=ValueError("cannot split")):
            with self.assertRaisesRegex(ValueError, "cannot split"):
                generate_splits(self.metadata_path, self.root / "failed")
        self.assertFalse((self.root / "failed").exists())
        self.assertEqual(list(self.root.glob(".failed.staging-*")), [])

    def test_cli_works_from_another_working_directory_and_refuses_rerun(self):
        output = self.root / "cli-output"
        command = [sys.executable, str(PROJECT / "scripts" / "generate_splits.py"),
                   "--metadata", str(self.metadata_path), "--output-dir", str(output)]
        first = subprocess.run(command, cwd=self.root, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = (output / "seed_2" / "test.csv").read_bytes()
        second = subprocess.run(command, cwd=self.root, text=True, capture_output=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn("Output already exists", second.stderr)
        self.assertEqual((output / "seed_2" / "test.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
