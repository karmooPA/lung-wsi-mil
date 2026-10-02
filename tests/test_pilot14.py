import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from scripts import run_14_wsi_pilot as pilot


class _FakeSlide:
    def close(self):
        return None


def _row(index, label, root):
    path = root / f"slide-{index}.svs"
    path.write_bytes(f"slide-{index}".encode("ascii"))
    return {
        "slide_id": f"TCGA-AA-{index:04d}-01A",
        "patient_id": f"TCGA-AA-{index:04d}",
        "label": label,
        "resolved_path": str(path),
        "duplicate": "False",
        "label_valid": "True",
        "expected_file_size_bytes": str(path.stat().st_size),
        "expected_md5": pilot._hash_file(path, "md5"),
    }


class PilotSelectionTests(unittest.TestCase):
    def test_input_output_and_repository_directories_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "repo"
            inputs = root / "inputs"
            outputs = root / "outputs"
            pilot._validate_run_directories(inputs, outputs, project)
            for output in (project, project / "results", root):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside the repository"):
                    pilot._validate_run_directories(inputs, output, project)
            for output in (inputs, inputs / "outputs"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "must not overlap"):
                    pilot._validate_run_directories(inputs, output, project)

    def test_pinned_cohort_matches_frozen_inventory(self):
        project = Path(__file__).resolve().parents[1]
        inventory = pilot._load_inventory(project / "metadata/dataset_inventory.csv")
        by_slide = {row["slide_id"]: row for row in inventory}
        with (project / "metadata/pilot14_selection.csv").open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))
        self.assertEqual(len(selected), 14)
        self.assertEqual(len({row["patient_id"] for row in selected}), 14)
        self.assertEqual(sum(row["label"] == "Normal" for row in selected), 7)
        self.assertEqual(sum(row["label"] == "Tumor" for row in selected), 7)
        for row in selected:
            for field, value in row.items():
                self.assertEqual(value, by_slide[row["slide_id"]][field])

    def test_balanced_selection_has_unique_patients_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [_row(i, "Normal" if i < 10 else "Tumor", root) for i in range(20)]
            selected = pilot._select_balanced(rows, seed=42)
            pilot._validate_selection(selected)
            self.assertEqual(len(selected), 14)
            self.assertEqual(len({row["patient_id"] for row in selected}), 14)
            self.assertEqual(len({row["resolved_path"] for row in selected}), 14)

    def test_selection_rejects_patient_and_physical_file_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = [_row(i, "Normal" if i < 7 else "Tumor", root) for i in range(14)]
            selected[1]["patient_id"] = selected[0]["patient_id"]
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                pilot._validate_selection(selected)
            selected[1]["patient_id"] = "TCGA-AA-9999"
            selected[1]["resolved_path"] = selected[0]["resolved_path"]
            with self.assertRaisesRegex(ValueError, "Duplicate physical WSI"):
                pilot._validate_selection(selected)

    def test_source_identity_checks_size_and_md5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = _row(1, "Normal", root)
            prepared = pilot._prepare_source_identities([row], verify_md5=True)[0]
            self.assertEqual(prepared["computed_md5"], row["expected_md5"])
            bad = dict(row, expected_file_size_bytes="999")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                pilot._prepare_source_identities([bad], verify_md5=False)
            bad = dict(row, expected_md5="0" * 32)
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                pilot._prepare_source_identities([bad], verify_md5=True)

    def test_inventory_relative_path_cannot_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            outside = Path(directory) / "outside.svs"
            outside.write_bytes(b"outside")
            row = {
                "slide_id": "TCGA-AA-0001-01A",
                "relative_path": "../outside.svs",
            }
            with self.assertRaisesRegex(ValueError, "escapes the dataset root"):
                pilot._resolve_inventory_row(row, root, {}, {})


class PilotStorageIntegrationTests(unittest.TestCase):
    def _result(self):
        return {
            "valid_patches": [
                {"x": 0, "y": 0, "tissue_percentage": 21.0},
                {"x": 2048, "y": 0, "tissue_percentage": 50.0},
                {"x": 4096, "y": 0, "tissue_percentage": 90.0},
            ],
            "status": "success",
            "candidate_tile_count": 3,
            "valid_patch_count": 3,
            "feature_extraction_ready": True,
            "resolution": {
                "target_magnification": 5.0,
                "target_downsample": 4.0,
                "level0_patch_size": 2048,
            },
            "thumbnail": Image.new("RGB", (8, 8), "white"),
            "filtered_mask": np.ones((8, 8), dtype=bool),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }

    @staticmethod
    def _patch(*, level0_x, **_kwargs):
        value = int(level0_x // 2048) * 50 + 40
        return Image.new("RGB", (512, 512), (value, 80, 140))

    def _save(self, root, name, chunk_size):
        output = root / name / "augmentation" / "patch_views.h5"
        identity = {
            "source_size_bytes": 10,
            "source_mtime_ns": 20,
            "expected_md5": "a" * 32,
            "computed_md5": "a" * 32,
        }
        with mock.patch.object(pilot, "open_slide", return_value=_FakeSlide()), mock.patch.object(
            pilot, "extract_patch", side_effect=self._patch
        ):
            metadata = pilot._save_augmented_views(
                slide_path="fake.svs",
                slide_id="TCGA-AA-0001-01A",
                patient_id="TCGA-AA-0001",
                label="Normal",
                result=self._result(),
                output_path=output,
                master_seed=42,
                chunk_size=chunk_size,
                save_qc=True,
                qc_sample_count=3,
                source_identity=identity,
                run_fingerprint="abc123",
            )
        return output, metadata

    def test_chunk_size_does_not_change_saved_views_and_qc_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one, first = self._save(root, "one", 1)
            two, second = self._save(root, "two", 2)
            with h5py.File(one, "r") as left, h5py.File(two, "r") as right:
                np.testing.assert_array_equal(left["images"][:], right["images"][:])
                np.testing.assert_array_equal(left["coords"][:], right["coords"][:])
            self.assertEqual(len(first["sha256"]), 64)
            self.assertEqual(len(second["sha256"]), 64)
            quality = one.parent / "augmentation_quality.csv"
            with quality.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue((one.parent / "augmentation_qc_montage.png").is_file())
            summary = json.loads((one.parent / "augmentation_quality_summary.json").read_text())
            self.assertEqual(summary["row_count"], 9)

    def test_resume_rejects_corrupt_hdf5_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5_path, metadata = self._save(root, "slide", 2)
            slide_dir = hdf5_path.parents[1]
            preprocessing = slide_dir / "preprocessing"
            source = root / "fake.svs"
            source.write_bytes(b"0123456789")
            config = pilot.build_config(5, 512, 20, 35, 1024, save_qc=True)
            pilot.save_slide_outputs(
                self._result(), source, preprocessing, config, 1.0, save_qc=True
            )
            manifest_row = {
                "slide_id": "TCGA-AA-0001-01A",
                "patient_id": "TCGA-AA-0001",
                "label": "Normal",
                "status": "success",
                "slide_path": str(source),
                "candidate_tile_count": "3",
                "valid_patch_count": "3",
                "augmentation_size_bytes": str(metadata["size_bytes"]),
                "augmentation_sha256": metadata["sha256"],
            }
            pilot._write_json_atomic(
                {"status": "success", "run_fingerprint": "abc123", "manifest_row": manifest_row},
                slide_dir / "complete.json",
            )
            self.assertIsNotNone(
                pilot._completed_payload(
                    slide_dir,
                    "abc123",
                    True,
                    preprocessing_config=config,
                    qc_sample_count=3,
                )
            )
            quality_path = hdf5_path.parent / "augmentation_quality.csv"
            quality = quality_path.read_text(encoding="utf-8")
            quality_path.write_text(quality.replace("0,0,0,", "0,1,0,", 1), encoding="utf-8")
            self.assertIsNone(
                pilot._completed_payload(
                    slide_dir, "abc123", True, preprocessing_config=config, qc_sample_count=3
                )
            )
            quality_path.write_text(quality, encoding="utf-8")
            (preprocessing / "qc" / "thumbnail.png").write_bytes(b"not-a-png")
            self.assertIsNone(
                pilot._completed_payload(
                    slide_dir, "abc123", True, preprocessing_config=config, qc_sample_count=3
                )
            )
            pilot.save_slide_outputs(
                self._result(), source, preprocessing, config, 1.0, save_qc=True
            )
            with hdf5_path.open("r+b") as handle:
                handle.truncate(128)
            self.assertIsNone(
                pilot._completed_payload(
                    slide_dir, "abc123", True, preprocessing_config=config, qc_sample_count=3
                )
            )

    def test_run_lock_and_atomic_directory_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = pilot._acquire_run_lock(root)
            with self.assertRaisesRegex(RuntimeError, "Another pilot run"):
                pilot._acquire_run_lock(root)
            lock.unlink()
            final = root / "final"
            working = root / "working"
            final.mkdir()
            working.mkdir()
            (final / "old").write_text("old", encoding="utf-8")
            (working / "new").write_text("new", encoding="utf-8")
            pilot._install_slide_directory(working, final)
            self.assertFalse((final / "old").exists())
            self.assertEqual((final / "new").read_text(), "new")


if __name__ == "__main__":
    unittest.main()
