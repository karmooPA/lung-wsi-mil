"""Run alongside the unchanged KimiaNet branch; never pretend to test weights.

These tests skip only if that branch's modules are absent. Dependency failures
on the combined checkout are errors, not skips. Use the narrow test pattern in
docs/data-augmentation.md to avoid executing the colleague's manual WSI demo.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lung_wsi_mil.augmentation import transform_batch, transform_patch

HAVE_FEATURES = importlib.util.find_spec("lung_wsi_mil.features.feature_extraction") is not None
if HAVE_FEATURES:
    import h5py

    from lung_wsi_mil.features.feature_extraction import extract_embeddings_with_coordinates
    from lung_wsi_mil.features.kimianet import KimiaNetFeatureExtractor
    from lung_wsi_mil.features.pipeline import process_transformed_slide
    from lung_wsi_mil.features.transforms import kimianet_transform


class MeanEncoder(torch.nn.Module):
    """Known-output stand-in tests routing/batching, not KimiaNet quality."""

    feature_dim = 1024

    def forward(self, tensor):
        scalar = tensor.mean(dim=(1, 2, 3))
        return scalar[:, None].expand(-1, self.feature_dim), torch.zeros(len(tensor), 30)


@unittest.skipUnless(HAVE_FEATURES, "requires feature/kimianet-features checkout")
class KimiaNetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.images = [
            Image.new("RGB", (512, 512), color)
            for color in [(120, 40, 90), (240, 170, 190), (40, 70, 160)]
        ]

    def test_original_matches_colleague_transform_exactly(self):
        original = transform_patch(self.images[0], rng=np.random.default_rng(4))[0]
        self.assertTrue(torch.equal(original, kimianet_transform(self.images[0])))

    def test_actual_batched_pipeline_coordinates_and_hdf5(self):
        transformed = transform_batch(self.images, rng=np.random.default_rng(9))
        coords = np.array([[4096, 4096], [8192, 4096], [8192, 8192]], dtype=np.int64)
        expected = transformed.mean(dim=(2, 3, 4)).numpy()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "slide.h5"
            arguments = dict(
                transformed_patches=transformed,
                coordinates=coords,
                model=MeanEncoder().eval(),
                device=torch.device("cpu"),
                hdf5_path=path,
                slide_id="synthetic",
                case_id="synthetic-patient",
                label=1,
                batch_size=2,
            )
            self.assertTrue(process_transformed_slide(**arguments))
            self.assertFalse(process_transformed_slide(**arguments))
            with h5py.File(path, "r") as f:
                self.assertEqual(f["synthetic/embeddings"].shape, (3, 3, 1024))
                self.assertEqual(f["synthetic/embeddings"].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(f["synthetic/coords"][:], coords)
                # Reduction kernels may round differently for the final short batch.
                np.testing.assert_allclose(
                    f["synthetic/embeddings"][:, :, 0], expected, rtol=1e-6, atol=1e-6
                )
                self.assertEqual(f["synthetic/label"][()], 1)
                self.assertEqual(f["synthetic/case_id"].asstr()[()], "synthetic-patient")

    def test_empty_batch_requires_shaped_coordinates(self):
        output = transform_batch([])
        embeddings, coords = extract_embeddings_with_coordinates(
            output, np.empty((0, 2), dtype=np.int64), MeanEncoder(), torch.device("cpu")
        )
        self.assertEqual(embeddings.shape, (3, 0, 1024))
        self.assertEqual(coords.shape, (0, 2))
        # Documents the existing caller obligation, without changing the colleague's code.
        with self.assertRaisesRegex(ValueError, "Coordinates must have shape"):
            extract_embeddings_with_coordinates(output, [], MeanEncoder(), torch.device("cpu"))

    def test_actual_untrained_kimianet_forward_accepts_patch(self):
        with torch.random.fork_rng():
            torch.manual_seed(11)
            model = KimiaNetFeatureExtractor().eval()
        original = transform_patch(self.images[0], rng=np.random.default_rng(2))[0]
        with torch.inference_mode():
            embeddings, logits = model(original.unsqueeze(0))
        self.assertEqual(embeddings.shape, (1, 1024))
        self.assertEqual(logits.shape, (1, 30))
        self.assertTrue(torch.isfinite(embeddings).all())


if __name__ == "__main__":
    unittest.main()
