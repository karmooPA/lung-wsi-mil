"""Portable checks of the frozen MIL HDF5 dataset contract."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset


class MILHDF5DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "features.h5"

    def write_slide(self, slide_id="slide_b", patch_count=3, **overrides):
        values = {
            "embeddings": np.arange(3 * patch_count * 1024, dtype=np.float32).reshape(
                3, patch_count, 1024
            )
            / 1024,
            "coords": np.arange(patch_count * 2, dtype=np.int64).reshape(patch_count, 2),
            "label": np.float32(1),
            "case_id": "patient_b",
        }
        values.update(overrides)
        with h5py.File(self.path, "a") as file:
            group = file.create_group(slide_id)
            for name, value in values.items():
                group.create_dataset(name, data=value)
        return values

    def test_round_trip_preserves_views_coordinates_and_identity(self):
        expected = self.write_slide()
        self.write_slide("slide_a", patch_count=1, label=np.float32(0), case_id="patient_a")
        dataset = MILHDF5Dataset(self.path)
        self.assertEqual(dataset.slide_ids, ["slide_a", "slide_b"])
        self.assertEqual(len(dataset), 2)
        features, coordinates, label, slide_id, case_id = dataset[1]
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(coordinates.dtype, torch.int64)
        self.assertEqual(label.dtype, torch.float32)
        self.assertEqual(label.ndim, 0)
        np.testing.assert_array_equal(features.numpy(), expected["embeddings"])
        np.testing.assert_array_equal(coordinates.numpy(), expected["coords"])
        self.assertEqual((label.item(), slide_id, case_id), (1.0, "slide_b", "patient_b"))

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            MILHDF5Dataset(self.path)

    def test_missing_required_dataset_is_reported(self):
        self.write_slide()
        with h5py.File(self.path, "a") as file:
            del file["slide_b/coords"]
        with self.assertRaisesRegex(KeyError, "coords"):
            MILHDF5Dataset(self.path)[0]

    def test_invalid_shapes_and_labels_are_rejected(self):
        invalid = [
            ({"embeddings": np.zeros((3, 1024))}, "embeddings must be 3D"),
            ({"embeddings": np.zeros((2, 3, 1024))}, "expected 3 embedding views"),
            ({"embeddings": np.zeros((3, 3, 8))}, "expected embedding dimension"),
            ({"coords": np.zeros((3, 3))}, "expected coords shape"),
            ({"coords": np.zeros((2, 2))}, "count mismatch"),
            ({"label": np.array([1])}, "label must be a scalar"),
            ({"label": np.float32(2)}, "label must be 0 or 1"),
        ]
        for overrides, message in invalid:
            with self.subTest(message=message):
                if self.path.exists():
                    self.path.unlink()
                self.write_slide(**overrides)
                with self.assertRaisesRegex(ValueError, message):
                    MILHDF5Dataset(self.path)[0]


if __name__ == "__main__":
    unittest.main()
