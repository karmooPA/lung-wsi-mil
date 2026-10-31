"""Offline regression tests. All image fixtures here are synthetic, not WSI."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image, ImageFilter

from lung_wsi_mil.augmentation import data_augmentations as aug
from lung_wsi_mil.augmentation.preview import make_preview
from lung_wsi_mil.preprocessing.patch_extraction import extract_patch


def synthetic_patch():
    y, x = np.indices((512, 512))
    rgb = np.stack(
        ((x // 16 % 2) * 180 + 30, (y // 32 % 2) * 120 + 60, (x + y) % 200 + 25), axis=-1
    ).astype(np.uint8)
    return Image.fromarray(rgb)


class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.image = synthetic_patch()

    def test_original_numeric_normalization_and_input_unchanged(self):
        before = self.image.tobytes()
        actual = aug.transform_patch(self.image, rng=np.random.default_rng(12))[0]
        pixels = np.asarray(self.image).astype(np.float32) / 255
        expected = (pixels - np.array(aug.IMAGENET_MEAN, dtype=np.float32)) / np.array(
            aug.IMAGENET_STD, dtype=np.float32
        )
        np.testing.assert_allclose(actual.permute(1, 2, 0).numpy(), expected, rtol=0, atol=0)
        self.assertEqual(self.image.tobytes(), before)

    def test_reproducible_independent_views_without_global_rng_changes(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        first = aug.transform_patch(self.image, rng=np.random.default_rng(42))
        second = aug.transform_patch(self.image, rng=np.random.default_rng(42))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(a.shape, (3, 512, 512))
            self.assertEqual(a.dtype, torch.float32)
            self.assertEqual(a.device.type, "cpu")
            self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.equal(first[1], first[2]))
        self.assertFalse(torch.equal(first[0], first[1]))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_rng_advances_across_calls_and_batch_keeps_patch_view_order(self):
        images = [self.image, Image.new("RGB", (512, 512), (120, 40, 90)), self.image]
        rng = np.random.default_rng(51)
        singles = [aug.transform_patch(p, rng=rng) for p in images]
        actual = aug.transform_batch(images, rng=np.random.default_rng(51))
        self.assertEqual(actual.shape, (3, 3, 3, 512, 512))
        for index in range(3):
            for version in range(3):
                self.assertTrue(torch.equal(actual[version, index], singles[index][version]))
        self.assertFalse(torch.equal(actual[1, 0], actual[1, 2]))

    def test_empty_and_singleton_batch(self):
        self.assertEqual(aug.transform_batch([]).shape, (3, 0, 3, 512, 512))
        output = aug.transform_batch([self.image], rng=np.random.default_rng(1))
        self.assertEqual(output.shape, (3, 1, 3, 512, 512))
        self.assertEqual(output.dtype, torch.float32)

    def test_invalid_input_rejected_before_random_state_advances(self):
        rng = np.random.default_rng(4)
        before = copy.deepcopy(rng.bit_generator.state)
        for bad in [
            np.zeros((512, 512, 3)),
            Image.new("RGBA", (512, 512)),
            Image.new("L", (512, 512)),
            Image.new("RGB", (256, 512)),
        ]:
            with self.subTest(type=type(bad), mode=getattr(bad, "mode", None)):
                with self.assertRaises((TypeError, ValueError)):
                    aug.transform_batch([self.image, bad], rng=rng)
                with self.assertRaises((TypeError, ValueError)):
                    aug.transform_patch(bad, rng=rng)
                self.assertEqual(before, rng.bit_generator.state)

    def test_stain_uses_two_scalars_and_preserves_d_channel(self):
        rng = Mock()
        rng.normal.return_value = np.array([1.02, 0.98])
        hed = np.full((2, 2, 3), 0.1)
        converted = np.full((2, 2, 3), 0.5)
        with (
            patch.object(aug, "rgb2hed", return_value=hed),
            patch.object(aug, "hed2rgb", return_value=converted) as to_rgb,
        ):
            result = aug.RandomHEStain(rng)(Image.new("RGB", (2, 2)))
        rng.normal.assert_called_once_with(1.0, 0.02, size=2)
        np.testing.assert_allclose(to_rgb.call_args.args[0][0, 0], [0.102, 0.098, 0.1])
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (127, 127, 127))

    def test_stain_clips_rgb_and_black_white_are_finite(self):
        with patch.object(aug, "hed2rgb", return_value=np.array([[[-0.2, 1.5, 0.5]]])):
            pixel = aug.RandomHEStain(np.random.default_rng(9))(Image.new("RGB", (1, 1)))
            self.assertEqual(pixel.getpixel((0, 0)), (0, 255, 127))
        for color in [(0, 0, 0), (255, 255, 255)]:
            for view in aug.transform_patch(
                Image.new("RGB", (512, 512), color), rng=np.random.default_rng(3)
            ):
                self.assertTrue(torch.isfinite(view).all())

    def test_blur_signed_scalar_radius_and_compatibility_name(self):
        rng = Mock()
        for radius in [-0.75, 0.75, 0.0]:
            rng.normal.return_value = radius
            output = aug.RandomGaussianNoise(rng)(self.image)
            expected = self.image.filter(ImageFilter.GaussianBlur(abs(radius)))
            np.testing.assert_array_equal(np.asarray(output), np.asarray(expected))
        rng.normal.assert_called_with(0.0, 0.5)
        self.assertIsInstance(aug.RandomGaussianNoise(), aug.RandomGaussianBlur)

    def test_blur_cannot_add_noise_to_a_constant_image(self):
        rng = Mock()
        rng.normal.return_value = 1.0
        image = Image.new("RGB", (64, 64), (120, 40, 90))
        self.assertEqual(aug.RandomGaussianNoise(rng)(image).tobytes(), image.tobytes())

    def test_quarter_turns_keep_exact_pixels(self):
        rng = Mock()
        for turn in range(4):
            rng.integers.return_value = turn
            result = aug.RotateRandom(rng)(self.image)
            np.testing.assert_array_equal(
                np.asarray(result), np.rot90(np.asarray(self.image), turn)
            )
        rng.integers.assert_called_with(0, 4)

    def test_flip_probability_boundary_and_geometry(self):
        rng = Mock()
        for transform, axis in [(aug.RandomHorizontalFlip, 1), (aug.RandomVerticalFlip, 0)]:
            for draw in [0.0, 0.499999, 0.5, 0.999999]:
                rng.random.return_value = draw
                result = transform(rng)(self.image)
                expected = (
                    np.flip(np.asarray(self.image), axis=axis)
                    if draw < 0.5
                    else np.asarray(self.image)
                )
                np.testing.assert_array_equal(np.asarray(result), expected)

    def test_both_pipelines_use_required_order_with_fresh_draws(self):
        rng = Mock()
        rng.normal.side_effect = [np.array([1.02, 0.99]), 0.2, np.array([0.97, 1.01]), 0.4]
        rng.integers.side_effect = [1, 2]
        rng.random.side_effect = [0.1, 0.9, 0.9, 0.1]
        views = aug.transform_patch(self.image, rng=rng)
        self.assertEqual(rng.normal.call_count, 4)
        self.assertEqual(rng.integers.call_count, 2)
        self.assertEqual(rng.random.call_count, 4)
        self.assertFalse(torch.equal(views[1], views[2]))
        pipelines = aug.make_aug_transforms(np.random.default_rng(1))
        for pipeline in pipelines:
            self.assertEqual(
                [type(t).__name__ for t in pipeline.transforms],
                [
                    "RandomHEStain",
                    "RandomGaussianNoise",
                    "RotateRandom",
                    "RandomHorizontalFlip",
                    "RandomVerticalFlip",
                    "Compose",
                ],
            )
        self.assertIsNot(pipelines[0], pipelines[1])

    def test_real_extract_patch_function_with_synthetic_slide(self):
        calls = []

        class SyntheticSlide:
            level_downsamples = (1.0, 3.5)

            def get_best_level_for_downsample(self, value):
                return 1

            def read_region(self, location, level, size):
                calls.append((location, level, size))
                return synthetic_patch().resize(size).convert("RGBA")

        image = extract_patch(SyntheticSlide(), 4096, 4096, target_downsample=4, patch_size=512)
        self.assertEqual(calls, [((4096, 4096), 1, (585, 585))])
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(aug.transform_batch([image]).shape, (3, 1, 3, 512, 512))

    def test_preview_roundtrip_original_and_input_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            input_path = Path(d) / "synthetic.png"
            output = Path(d) / "qc.png"
            self.image.save(input_path)
            before = input_path.read_bytes()
            metadata = make_preview(input_path, output, seed=42)
            with Image.open(output) as panel:
                self.assertEqual(panel.size, (1536, 552))
                self.assertEqual(panel.crop((0, 40, 512, 552)).tobytes(), self.image.tobytes())
            self.assertEqual(before, input_path.read_bytes())
            self.assertEqual(metadata["input_name"], "synthetic.png")
            self.assertTrue(output.with_suffix(".json").is_file())
            with self.assertRaises(ValueError):
                make_preview(input_path, input_path, seed=42)


if __name__ == "__main__":
    unittest.main()
