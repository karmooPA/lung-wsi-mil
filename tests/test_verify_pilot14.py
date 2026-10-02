import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from scripts import run_14_wsi_pilot as pilot
from scripts import verify_pilot14 as verifier
from src.augmentation.data_augmentations import IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE


class _FakeSlide:
    dimensions = (8192, 8192)

    def close(self):
        return None


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SlideFixture:
    quality_fields = (
        "patch_index",
        "x",
        "y",
        "tissue_percentage",
        "representation",
        "mean_intensity",
        "std_intensity",
        "white_fraction",
        "dark_fraction",
        "clipped_fraction",
        "mean_saturation",
        "gradient_energy",
        "identical_to_original",
    )

    def __init__(
        self,
        root,
        *,
        patch_count=1,
        save_qc=True,
        qc_sample_count=1,
        aug1_identical=False,
        aug2_identical=False,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.slide_id = "TCGA-AA-0001-01A"
        self.patient_id = "TCGA-AA-0001"
        self.label = "Normal"
        self.source = self.root / "source.svs"
        self.source.write_bytes(b"synthetic-wsi")
        source_stat = self.source.stat()
        source_md5 = hashlib.md5(self.source.read_bytes()).hexdigest()
        self.identity = {
            "slide_id": self.slide_id,
            "patient_id": self.patient_id,
            "label": self.label,
            "resolved_path": str(self.source.resolve()),
            "source_size_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "expected_md5": source_md5,
            "computed_md5": source_md5,
        }
        self.preprocessing_config = pilot.build_config(
            target_magnification=5,
            patch_size=PATCH_SIZE,
            min_tissue_percentage=20,
            color_distance_threshold=35,
            thumbnail_max_size=1024,
            save_qc=save_qc,
        )
        self.run_config = {
            "run_fingerprint": "abc123",
            "save_qc": save_qc,
            "qc_sample_count": qc_sample_count,
            "preprocessing": self.preprocessing_config,
            "source_identities": [self.identity],
        }
        self.slide_dir = self.root / "slides" / self.slide_id
        preprocessing = self.slide_dir / "preprocessing"
        augmentation = self.slide_dir / "augmentation"
        preprocessing.mkdir(parents=True)
        augmentation.mkdir(parents=True)

        coordinates = np.asarray(
            [[index * PATCH_SIZE, 0] for index in range(patch_count)],
            dtype=np.int64,
        )
        if patch_count == 0:
            coordinates = np.empty((0, 2), dtype=np.int64)
        tissue = np.full(patch_count, 50.0, dtype=np.float32)
        np.save(preprocessing / "coordinates.npy", coordinates)
        _write_csv(
            preprocessing / "valid_patches.csv",
            ("x", "y", "tissue_percentage"),
            [
                {"x": int(x), "y": int(y), "tissue_percentage": "50.000000"}
                for x, y in coordinates
            ],
        )
        (preprocessing / "result.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "slide_path": str(self.source.resolve()),
                    "config": self.preprocessing_config,
                    "config_fingerprint": pilot.config_fingerprint(
                        self.preprocessing_config
                    ),
                    "source": {
                        "size_bytes": source_stat.st_size,
                        "mtime_ns": source_stat.st_mtime_ns,
                    },
                    "result": {
                        "candidate_tile_count": patch_count,
                        "valid_patch_count": patch_count,
                        "feature_extraction_ready": patch_count > 0,
                        "valid_patches": [
                            {
                                "x": int(x),
                                "y": int(y),
                                "tissue_percentage": 50.0,
                            }
                            for x, y in coordinates
                        ],
                        "resolution": {
                            "target_magnification": 5.0,
                            "target_downsample": 4.0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        y, x = np.indices((PATCH_SIZE, PATCH_SIZE))
        base = np.stack(
            (
                (x % 251),
                (y % 247),
                ((x + y) % 241),
            ),
            axis=-1,
        ).astype(np.uint8)
        originals = np.stack(
            [np.roll(base, index, axis=0) for index in range(patch_count)],
            axis=0,
        ) if patch_count else np.empty((0, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        aug1 = originals.copy() if aug1_identical else np.roll(originals, 1, axis=2)
        aug2 = originals.copy() if aug2_identical else (255 - originals)
        self.images = np.stack((originals, aug1, aug2), axis=0)
        self.original_image = Image.fromarray(base)

        self.hdf5_path = augmentation / "patch_views.h5"
        with h5py.File(self.hdf5_path, "w") as file:
            file.create_dataset("images", data=self.images)
            file.create_dataset("coords", data=coordinates)
            file.create_dataset("tissue_percentage", data=tissue)
            file.attrs.update(
                format_version="2.0",
                run_fingerprint="abc123",
                slide_id=self.slide_id,
                patient_id=self.patient_id,
                label=self.label,
                representations=json.dumps(list(verifier.REPRESENTATIONS)),
                image_layout="representation,patch,height,width,channel",
                stored_dtype="uint8",
                restored_tensor_layout="representation,patch,channel,height,width",
                restored_tensor_dtype="float32",
                imagenet_mean=json.dumps(list(IMAGENET_MEAN)),
                imagenet_std=json.dumps(list(IMAGENET_STD)),
                patch_size=PATCH_SIZE,
                patch_count=patch_count,
                source_path=str(self.source.resolve()),
                target_magnification=5.0,
                target_downsample=4.0,
                source_size_bytes=source_stat.st_size,
                source_mtime_ns=source_stat.st_mtime_ns,
                source_expected_md5=source_md5,
                source_computed_md5=source_md5,
            )

        quality_rows = pilot._quality_rows(self.images, coordinates, tissue, 0)
        _write_csv(
            augmentation / "augmentation_quality.csv",
            self.quality_fields,
            quality_rows,
        )
        (augmentation / "augmentation_quality_summary.json").write_text(
            json.dumps(pilot._quality_summary(quality_rows)),
            encoding="utf-8",
        )

        if save_qc:
            qc = preprocessing / "qc"
            qc.mkdir()
            for name in ("thumbnail.png", "filtered_mask.png", "valid_patches_overlay.png"):
                Image.new("RGB", (8, 8), "white").save(qc / name)
            if patch_count:
                Image.new("RGB", (8, 8), "white").save(
                    augmentation / "augmentation_qc_patch_0000.png"
                )
                if qc_sample_count:
                    Image.new("RGB", (8, 8), "white").save(
                        augmentation / "augmentation_qc_montage.png"
                    )

        self.row = {
            "slide_id": self.slide_id,
            "patient_id": self.patient_id,
            "label": self.label,
            "status": "success",
            "slide_path": str(self.source.resolve()),
            "candidate_tile_count": str(patch_count),
            "valid_patch_count": str(patch_count),
            "feature_extraction_ready": str(patch_count > 0),
            "augmentation_hdf5": str(self.hdf5_path.resolve()),
            "augmentation_size_bytes": str(self.hdf5_path.stat().st_size),
            "augmentation_sha256": verifier._hash_file(self.hdf5_path),
            "source_size_bytes": str(source_stat.st_size),
            "source_md5": source_md5,
        }
        self._write_complete()

    def _write_complete(self):
        (self.slide_dir / "complete.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "run_fingerprint": "abc123",
                    "source_identity": self.identity,
                    "manifest_row": self.row,
                }
            ),
            encoding="utf-8",
        )

    def refresh_hdf5_digest(self):
        self.row["augmentation_size_bytes"] = str(self.hdf5_path.stat().st_size)
        self.row["augmentation_sha256"] = verifier._hash_file(self.hdf5_path)
        self._write_complete()

    def verify(self):
        with mock.patch.object(verifier, "open_slide", return_value=_FakeSlide()), mock.patch.object(
            verifier,
            "extract_patch",
            return_value=self.original_image,
        ):
            return verifier._verify_slide(
                self.row,
                self.root,
                self.run_config,
                verify_source_md5=False,
                source_sample_count=1,
            )


class PilotVerifierTests(unittest.TestCase):
    def test_valid_qc_run_and_no_qc_zero_patch_run(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = SlideFixture(Path(directory) / "with-qc")
            self.assertEqual(valid.verify()["patch_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            empty = SlideFixture(
                Path(directory) / "empty-no-qc",
                patch_count=0,
                save_qc=False,
                qc_sample_count=0,
            )
            self.assertEqual(empty.verify()["patch_count"], 0)

    def test_either_fully_identical_augmentation_stream_fails(self):
        for field in ("aug1_identical", "aug2_identical"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = SlideFixture(Path(directory), **{field: True}, save_qc=False)
                with self.assertRaisesRegex(ValueError, "All Aug[12] views equal"):
                    fixture.verify()

    def test_hdf5_patient_and_completion_source_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            with h5py.File(fixture.hdf5_path, "r+") as file:
                file.attrs["patient_id"] = "TCGA-WRONG"
            fixture.refresh_hdf5_digest()
            with self.assertRaisesRegex(ValueError, "HDF5 patient identity mismatch"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            complete = json.loads(
                (fixture.slide_dir / "complete.json").read_text(encoding="utf-8")
            )
            complete["source_identity"]["source_size_bytes"] += 1
            (fixture.slide_dir / "complete.json").write_text(
                json.dumps(complete), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Completion source identity"):
                fixture.verify()

    def test_preprocessing_manifest_contract_is_cross_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            fixture.row["candidate_tile_count"] = "2"
            fixture._write_complete()
            with self.assertRaisesRegex(ValueError, "candidate tile count mismatch"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            fixture.row["feature_extraction_ready"] = "False"
            fixture._write_complete()
            with self.assertRaisesRegex(ValueError, "feature_extraction_ready mismatch"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            fixture.row["augmentation_hdf5"] = str(
                (fixture.root / "wrong" / "patch_views.h5").resolve()
            )
            fixture._write_complete()
            with self.assertRaisesRegex(ValueError, "augmentation_hdf5 path mismatch"):
                fixture.verify()

    def test_saved_preprocessing_config_and_source_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            path = fixture.slide_dir / "preprocessing" / "result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["config"]["min_tissue_percentage"] = 10.0
            payload["config_fingerprint"] = pilot.config_fingerprint(payload["config"])
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requested run"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            path = fixture.slide_dir / "preprocessing" / "result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["slide_path"] = str((fixture.root / "other.svs").resolve())
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source path"):
                fixture.verify()

    def test_hdf5_source_path_and_resolution_match_preprocessing(self):
        cases = (
            ("source_path", "/wrong/source.svs", "HDF5 source path mismatch"),
            (
                "target_magnification",
                10.0,
                "HDF5/preprocessing target magnification mismatch",
            ),
            (
                "target_downsample",
                8.0,
                "HDF5/preprocessing target downsample mismatch",
            ),
        )
        for attribute, value, message in cases:
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as directory:
                fixture = SlideFixture(Path(directory), save_qc=False)
                with h5py.File(fixture.hdf5_path, "r+") as file:
                    file.attrs[attribute] = value
                fixture.refresh_hdf5_digest()
                with self.assertRaisesRegex(ValueError, message):
                    fixture.verify()

    def test_quality_csv_and_summary_are_recomputed_from_hdf5(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            path = fixture.slide_dir / "augmentation" / "augmentation_quality.csv"
            rows = verifier._read_csv(path)
            rows[0]["x"] = "999"
            _write_csv(path, fixture.quality_fields, rows)
            with self.assertRaisesRegex(ValueError, "Quality row 0 x mismatch"):
                fixture.verify()

        with tempfile.TemporaryDirectory() as directory:
            fixture = SlideFixture(Path(directory), save_qc=False)
            path = fixture.slide_dir / "augmentation" / "augmentation_quality_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["representations"]["Original"]["count"] = 99
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Quality summary mismatch"):
                fixture.verify()

    def test_control_files_are_fully_cross_checked(self):
        selection = []
        manifest = []
        identities = []
        for index in range(14):
            label = "Normal" if index < 7 else "Tumor"
            slide_id = f"TCGA-AA-{index:04d}-01A"
            patient_id = f"TCGA-AA-{index:04d}"
            source = str((Path("/data") / f"{slide_id}.svs").resolve())
            identity = {
                "slide_id": slide_id,
                "patient_id": patient_id,
                "label": label,
                "resolved_path": source,
                "source_size_bytes": 100 + index,
                "source_mtime_ns": 1000 + index,
                "expected_md5": f"{index:032x}",
                "computed_md5": f"{index:032x}",
            }
            identities.append(identity)
            selection.append({key: str(value) for key, value in identity.items()})
            manifest.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "label": label,
                    "slide_path": source,
                    "source_size_bytes": str(100 + index),
                    "source_md5": f"{index:032x}",
                    "status": "success",
                }
            )
        run_config = {
            "selected_slide_ids": [row["slide_id"] for row in selection],
            "source_identities": identities,
            "save_qc": True,
        }
        run_config["run_fingerprint"] = verifier._recompute_run_fingerprint(run_config)
        self.assertEqual(
            verifier._validate_control_files(selection, manifest, run_config),
            [],
        )

        changed = copy.deepcopy(manifest)
        changed[0]["patient_id"] = "TCGA-WRONG"
        errors = verifier._validate_control_files(selection, changed, run_config)
        self.assertTrue(any("patient_id" in error for error in errors))

        changed_config = copy.deepcopy(run_config)
        changed_config["selected_slide_ids"] = list(reversed(changed_config["selected_slide_ids"]))
        errors = verifier._validate_control_files(selection, manifest, changed_config)
        self.assertTrue(any("selected_slide_ids" in error for error in errors))
        self.assertTrue(any("fingerprint" in error for error in errors))

    def test_verify_reports_pipeline_code_mismatch_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_config = {
                "run_fingerprint": "test-run",
                "pipeline_code_sha256": "a" * 64,
            }
            (root / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            selection = []
            manifest = []
            for index in range(14):
                label = "Normal" if index < 7 else "Tumor"
                slide_id = f"TCGA-ZZ-{index:04d}-01A"
                selection.append({"slide_id": slide_id})
                manifest.append(
                    {
                        "slide_id": slide_id,
                        "patient_id": f"TCGA-ZZ-{index:04d}",
                        "label": label,
                        "slide_path": str(root / f"{slide_id}.svs"),
                        "status": "success",
                    }
                )
            _write_csv(root / "selection.csv", ("slide_id",), selection)
            _write_csv(
                root / "manifest.csv",
                ("slide_id", "patient_id", "label", "slide_path", "status"),
                manifest,
            )

            def verified_detail(row, *_args, **_kwargs):
                return {
                    "slide_id": row["slide_id"],
                    "label": row["label"],
                    "patch_count": 0,
                    "hdf5_bytes": 0,
                    "source_reextract_samples": 0,
                }

            with mock.patch.object(
                verifier, "_code_fingerprint", return_value="b" * 64
            ), mock.patch.object(
                verifier, "_validate_control_files", return_value=[]
            ), mock.patch.object(
                verifier, "_verify_slide", side_effect=verified_detail
            ):
                report = verifier.verify(root, source_sample_count=0)

            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["current_pipeline_code_sha256"], "b" * 64)
            self.assertTrue(
                any(
                    "Current pipeline code SHA256 does not match" in error
                    and "expected" in error
                    and "observed" in error
                    for error in report["errors"]
                )
            )
            saved = json.loads(
                (root / "verification.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
