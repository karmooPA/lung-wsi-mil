"""Cheap encoders test streaming/alignment; these do not validate KimiaNet weights."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
from lung_wsi_mil.features.hdf5_pipeline import extract_feature_cache, hash_file


class ChannelEncoder(torch.nn.Module):
    feature_dim = 1024

    def __init__(self, fail=False, nonfinite=False):
        super().__init__()
        self.calls = []
        self.fail = fail
        self.nonfinite = nonfinite
        self.eval()

    def forward(self, images):
        self.calls.append(tuple(images.shape))
        if self.fail:
            raise RuntimeError("Injected encoder failure")
        # Fixtures are spatially constant RGB; independent expectations below
        # can therefore identify channel, view, and patch routing exactly.
        result = torch.empty(len(images), 1024, device=images.device)
        result[:] = images[:, 0, 0, 0, None]
        result[:, :3] = images[:, :, 0, 0]
        if self.nonfinite:
            result[0, 0] = torch.nan
        return result, torch.zeros(len(images), 30, device=images.device)


class FeatureHDF5PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inventory = self.root / "inventory.csv"
        self.selection = self.root / "selection.csv"
        self.weights = self.root / "mock-unit-test-weights.bin"
        self.weights.write_bytes(b"not-a-real-model-unit-tests-patch-loader")
        self.output = self.root / "features.h5"
        self.rows = [
            {"slide_id": f"slide_{i}", "patient_id": f"case_{i}", "label": label}
            for i, label in enumerate(("Normal", "Tumor"))
        ]
        self._csv(self.inventory, self.rows)
        self._csv(self.selection, self.rows[::-1])
        self.images = {}
        self.paths = {}
        for index, n in enumerate((3, 1)):
            pixels = np.empty((3, n, 512, 512, 3), dtype=np.uint8)
            for view in range(3):
                for row in range(n):
                    pixels[view, row] = [
                        20 + index * 7 + view * 30 + row,
                        80 + view * 20 + row,
                        180 - view * 10 - row,
                    ]
            self._source(index, pixels)

    @staticmethod
    def _csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("slide_id", "patient_id", "label"))
            writer.writeheader()
            writer.writerows(rows)

    def _source(self, index, pixels):
        row = self.rows[index]
        n = pixels.shape[1]
        path = self.root / "slides" / row["slide_id"] / "augmentation/patch_views.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as file:
            file.create_dataset("images", data=pixels)
            file.create_dataset(
                "coords",
                data=np.column_stack((np.arange(n) * 512, np.arange(n) * 256)).astype(np.int64),
            )
            file.create_dataset("tissue_percentage", data=np.full(n, 55, dtype=np.float32))
            file.attrs.update(
                format_version="2.0",
                slide_id=row["slide_id"],
                patient_id=row["patient_id"],
                label=row["label"],
                representations=json.dumps(["Original", "Aug1", "Aug2"]),
                image_layout="representation,patch,height,width,channel",
                stored_dtype="uint8",
                restored_tensor_layout="representation,patch,channel,height,width",
                restored_tensor_dtype="float32",
                imagenet_mean=json.dumps([0.485, 0.456, 0.406]),
                imagenet_std=json.dumps([0.229, 0.224, 0.225]),
                patch_size=512,
                patch_count=n,
                run_fingerprint="fixture-pipeline-v1",
            )
        self.images[row["slide_id"]] = pixels
        self.paths[row["slide_id"]] = path

    def _run(self, model=None, **kwargs):
        model = ChannelEncoder() if model is None else model
        arguments = dict(
            patch_root=self.root,
            inventory=self.inventory,
            selection=self.selection,
            weights=self.weights,
            output=self.output,
            batch_size=2,
            device="cpu",
        )
        arguments.update(kwargs)
        with (
            patch("lung_wsi_mil.features.hdf5_pipeline._load_encoder", return_value=model),
            redirect_stdout(io.StringIO()),
        ):
            return extract_feature_cache(**arguments)

    def test_streams_all_views_preserves_alignment_and_normalization(self):
        encoder = ChannelEncoder()
        report = self._run(encoder)
        self.assertEqual((report["slides"], report["patches"], report["patch_views"]), (2, 4, 12))
        self.assertEqual(report["status"], "created")
        self.assertEqual(len(encoder.calls), 9)
        self.assertTrue(
            all(shape[0] <= 2 and shape[1:] == (3, 512, 512) for shape in encoder.calls)
        )
        dataset = MILHDF5Dataset(self.output)
        for index in range(len(dataset)):
            embeddings, coords, label, slide_id, case_id = dataset[index]
            pixels = self.images[slide_id][:, :, 0, 0].astype(np.float32)
            expected = (
                pixels / np.float32(255) - np.array([0.485, 0.456, 0.406], np.float32)
            ) / np.array([0.229, 0.224, 0.225], np.float32)
            np.testing.assert_array_equal(embeddings[:, :, :3].numpy(), expected)
            np.testing.assert_array_equal(embeddings[:, :, 3].numpy(), expected[:, :, 0])
            self.assertEqual(case_id, "case_" + slide_id[-1])
            self.assertEqual(label.item(), int(slide_id[-1]))
            np.testing.assert_array_equal(coords[:, 0].numpy(), np.arange(len(coords)) * 512)
        with h5py.File(self.output, "r") as file:
            self.assertEqual(file.attrs["weights_sha256"], hash_file(self.weights))
            provenance = json.loads(file.attrs["provenance_json"])
            self.assertEqual(provenance["config"]["representations"], ["Original", "Aug1", "Aug2"])
            self.assertEqual(len(provenance["sources"]), 2)
        self.assertFalse(list(self.root.glob("features.h5.partial.*")))
        self.assertFalse(self.output.with_name(self.output.name + ".lock").exists())

    def test_resume_identical_complete_cache_does_not_call_encoder(self):
        self._run()
        before = hash_file(self.output)
        with (
            patch(
                "lung_wsi_mil.features.hdf5_pipeline._load_encoder",
                side_effect=AssertionError("should not load"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            report = extract_feature_cache(
                patch_root=self.root,
                inventory=self.inventory,
                selection=self.selection,
                weights=self.weights,
                output=self.output,
                batch_size=2,
                device="cpu",
                resume=True,
            )
        self.assertEqual(report["status"], "resumed")
        self.assertEqual(hash_file(self.output), before)
        with self.assertRaises(FileExistsError):
            self._run()

    def test_changed_weights_source_or_configuration_refuse_stale_cache(self):
        self._run()
        original = hash_file(self.output)
        original_weights = self.weights.read_bytes()
        original_source = self.paths["slide_0"].read_bytes()
        for change in ("weights", "source", "batch_size"):
            with self.subTest(change=change):
                self.weights.write_bytes(original_weights)
                self.paths["slide_0"].write_bytes(original_source)
                if change == "weights":
                    self.weights.write_bytes(b"changed weights")
                elif change == "source":
                    with h5py.File(self.paths["slide_0"], "r+") as source:
                        source["images"][0, 0, 0, 0, 0] += 1
                with self.assertRaisesRegex(ValueError, "Stale"):
                    self._run(resume=True, **({"batch_size": 1} if change == "batch_size" else {}))
                self.assertEqual(hash_file(self.output), original)

    def test_resume_detects_nonfinite_finite_and_metadata_corruption(self):
        self._run()
        valid = self.output.read_bytes()
        for change in ("nan", "finite", "coords", "case_id", "missing_group"):
            with self.subTest(change=change):
                self.output.write_bytes(valid)
                with h5py.File(self.output, "r+") as target:
                    if change in {"nan", "finite"}:
                        target["slide_0/embeddings"][0, 0, 0] = np.nan if change == "nan" else 999
                    elif change == "coords":
                        target["slide_0/coords"][0, 0] = 999
                    elif change == "case_id":
                        del target["slide_0/case_id"]
                        target["slide_0"].create_dataset("case_id", data=b"wrong-case")
                    else:
                        del target["slide_0"]
                corrupted = hash_file(self.output)
                with self.assertRaises(ValueError):
                    self._run(resume=True)
                self.assertEqual(hash_file(self.output), corrupted)

    def test_model_failure_or_nonfinite_features_cannot_publish_partial_cache(self):
        for model in (ChannelEncoder(fail=True), ChannelEncoder(nonfinite=True)):
            with self.subTest(model=model), self.assertRaises((RuntimeError, ValueError)):
                self._run(model)
            self.assertFalse(self.output.exists())
            self.assertFalse(list(self.root.glob("features.h5.partial.*")))
            self.assertFalse(self.output.with_name(self.output.name + ".lock").exists())

    def test_empty_bag_and_duplicate_selection_are_explicit_errors(self):
        self._csv(self.selection, [self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._run()
        self._csv(self.selection, self.rows)
        self._source(0, np.empty((3, 0, 512, 512, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "empty bag"):
            self._run()
        self.assertFalse(self.output.exists())

    def test_inventory_mismatch_is_rejected(self):
        self._csv(self.selection, [{**self.rows[0], "label": "Tumor"}])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self._run()
        self._csv(self.selection, self.rows)
        with h5py.File(self.paths["slide_0"], "r+") as source:
            source.attrs["patient_id"] = "wrong-patient"
        with self.assertRaisesRegex(ValueError, "patient_id"):
            self._run()


if __name__ == "__main__":
    unittest.main()
