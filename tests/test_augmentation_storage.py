import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from lung_wsi_mil.augmentation import transform_batch
from lung_wsi_mil.augmentation.storage import (
    load_augmented_chunk,
    normalized_views_to_uint8,
    uint8_views_to_normalized,
    validate_augmented_hdf5,
)


class AugmentationStorageTests(unittest.TestCase):
    @staticmethod
    def _write_contract(file, stored, coords, tissue):
        file.create_dataset("images", data=stored)
        file.create_dataset("coords", data=coords)
        file.create_dataset("tissue_percentage", data=tissue)
        file.attrs.update(
            format_version="2.0",
            slide_id="slide-1",
            patient_id="patient-1",
            label="Normal",
            representations=json.dumps(["Original", "Aug1", "Aug2"]),
            image_layout="representation,patch,height,width,channel",
            stored_dtype="uint8",
            restored_tensor_layout="representation,patch,channel,height,width",
            restored_tensor_dtype="float32",
            imagenet_mean=json.dumps([0.485, 0.456, 0.406]),
            imagenet_std=json.dumps([0.229, 0.224, 0.225]),
            patch_size=512,
            patch_count=stored.shape[1],
            run_fingerprint="abc",
        )

    def test_uint8_storage_roundtrip_is_bit_exact(self):
        images = [
            Image.new("RGB", (512, 512), (120, 40, 90)),
            Image.new("RGB", (512, 512), (240, 170, 190)),
        ]
        transformed = transform_batch(images, rng=np.random.default_rng(123))
        stored = normalized_views_to_uint8(transformed)
        restored = uint8_views_to_normalized(stored)
        self.assertEqual(stored.shape, (3, 2, 512, 512, 3))
        self.assertEqual(stored.dtype, np.uint8)
        self.assertTrue(torch.equal(transformed, restored))

    def test_hdf5_chunk_loader_restores_contract(self):
        images = [Image.new("RGB", (512, 512), (30, 100, 200)) for _ in range(3)]
        transformed = transform_batch(images, rng=np.random.default_rng(9))
        stored = normalized_views_to_uint8(transformed)
        coords = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
        tissue = np.array([30.0, 40.0, 50.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.h5"
            with h5py.File(path, "w") as file:
                self._write_contract(file, stored, coords, tissue)

            loaded, loaded_coords, loaded_tissue = load_augmented_chunk(path, 1, 3)
            metadata = validate_augmented_hdf5(
                path,
                expected={"slide_id": "slide-1", "patch_count": 3, "run_fingerprint": "abc"},
            )

        self.assertTrue(torch.equal(loaded, transformed[:, 1:3]))
        np.testing.assert_array_equal(loaded_coords, coords[1:3])
        np.testing.assert_array_equal(loaded_tissue, tissue[1:3])
        self.assertEqual(metadata["label"], "Normal")

    def test_loader_rejects_misaligned_datasets_and_metadata(self):
        stored = np.zeros((3, 2, 512, 512, 3), dtype=np.uint8)
        coords = np.zeros((1, 2), dtype=np.int64)
        tissue = np.zeros(2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misaligned.h5"
            with h5py.File(path, "w") as file:
                self._write_contract(file, stored, coords, tissue)
            with self.assertRaisesRegex(ValueError, "coords"):
                load_augmented_chunk(path)

            path = Path(directory) / "bad-metadata.h5"
            with h5py.File(path, "w") as file:
                self._write_contract(
                    file,
                    stored,
                    np.zeros((2, 2), dtype=np.int64),
                    tissue,
                )
                file.attrs["patch_count"] = 99
            with self.assertRaisesRegex(ValueError, "patch_count"):
                validate_augmented_hdf5(path)

    def test_loader_rejects_invalid_slice_and_nonfinite_tissue(self):
        stored = np.zeros((3, 1, 512, 512, 3), dtype=np.uint8)
        coords = np.zeros((1, 2), dtype=np.int64)
        tissue = np.array([np.nan], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.h5"
            with h5py.File(path, "w") as file:
                self._write_contract(file, stored, coords, tissue)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_augmented_hdf5(path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_augmented_chunk(path)
            with h5py.File(path, "r+") as file:
                file["coords"][:] = (-1, 2)
                file["tissue_percentage"][:] = 50.0
            with self.assertRaisesRegex(ValueError, "negative"):
                load_augmented_chunk(path)
            with h5py.File(path, "r+") as file:
                file["coords"][:] = 0
            with self.assertRaisesRegex(ValueError, "Invalid slice"):
                load_augmented_chunk(path, 1, 0)

            with h5py.File(path, "r+") as file:
                file["tissue_percentage"][:] = np.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_augmented_chunk(path)

    def test_storage_rejects_wrong_contract(self):
        with self.assertRaises(ValueError):
            normalized_views_to_uint8(torch.zeros(3, 1, 3, 256, 256))
        with self.assertRaises(ValueError):
            uint8_views_to_normalized(np.zeros((3, 1, 512, 512, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
