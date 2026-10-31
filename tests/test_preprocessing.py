import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import openslide
from PIL import Image

from lung_wsi_mil.preprocessing.batch_preprocessing import (
    _acquire_output_lock,
    build_config,
    load_completed_result,
    save_slide_error,
    save_slide_outputs,
    slide_output_dir,
)
from lung_wsi_mil.preprocessing.patch_coordinates import (
    get_patch_coordinates,
    tile_indices_to_level0_coordinates,
)
from lung_wsi_mil.preprocessing.patch_extraction import (
    calculate_patch_tissue_percentage,
    extract_patch,
    is_valid_patch,
)
from lung_wsi_mil.preprocessing.pipeline import preprocess_slide
from lung_wsi_mil.preprocessing.wsi_reader import (
    estimate_objective_power_from_mpp,
    resolve_target_resolution,
)


class SyntheticSlide:
    def __init__(self, *, objective=20.0, mpp_x=0.5, mpp_y=0.5):
        self.dimensions = (4096, 4096)
        self.level_count = 2
        self.level_dimensions = ((4096, 4096), (1024, 1024))
        self.level_downsamples = (1.0, 4.0)
        self.properties = {
            openslide.PROPERTY_NAME_OBJECTIVE_POWER: str(objective),
            openslide.PROPERTY_NAME_MPP_X: str(mpp_x),
            openslide.PROPERTY_NAME_MPP_Y: str(mpp_y),
            openslide.PROPERTY_NAME_VENDOR: "synthetic",
        }
        self.level_one = Image.new("RGB", (1024, 1024), "white")
        pixels = np.asarray(self.level_one).copy()
        pixels[128:896, 160:864] = (165, 70, 130)
        pixels[300:700, 300:700] = (105, 35, 100)
        self.level_one = Image.fromarray(pixels)
        self.closed = False

    def get_thumbnail(self, size):
        result = self.level_one.copy()
        result.thumbnail(size)
        return result

    def get_best_level_for_downsample(self, downsample):
        return 1 if downsample >= 2 else 0

    def read_region(self, location, level, size):
        scale = self.level_downsamples[level]
        x = int(location[0] / scale)
        y = int(location[1] / scale)
        source = self.level_one if level == 1 else self.level_one.resize(self.dimensions)
        crop = source.crop((x, y, x + size[0], y + size[1])).convert("RGBA")
        if crop.size != size:
            canvas = Image.new("RGBA", size, (255, 255, 255, 0))
            canvas.paste(crop, (0, 0))
            crop = canvas
        return crop

    def close(self):
        self.closed = True


class ResolutionTests(unittest.TestCase):
    def test_agreeing_objective_and_mpp_resolve_exact_scale(self):
        slide = SyntheticSlide()
        info = {"objective_power": 20.0, "mpp_x": 0.5, "mpp_y": 0.5}
        result = resolve_target_resolution(slide, info, 5.0)
        self.assertEqual(result["source_magnification_method"], "objective_power_validated_by_mpp")
        self.assertEqual(result["target_downsample"], 4.0)
        self.assertEqual(result["target_mpp_x"], 2.0)

    def test_resolution_rejects_conflict_anisotropy_and_nonfinite_values(self):
        slide = SyntheticSlide()
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_target_resolution(
                slide,
                {"objective_power": 40.0, "mpp_x": 0.5, "mpp_y": 0.5},
                5.0,
            )
        with self.assertRaisesRegex(ValueError, "anisotropy"):
            estimate_objective_power_from_mpp(0.25, 0.5)
        self.assertIsNone(estimate_objective_power_from_mpp(math.inf, math.nan))


class PipelineTests(unittest.TestCase):
    def test_end_to_end_synthetic_slide_closes_and_returns_valid_coordinates(self):
        slide = SyntheticSlide()
        with mock.patch("lung_wsi_mil.preprocessing.pipeline.open_slide", return_value=slide):
            result = preprocess_slide("synthetic.svs", thumbnail_max_size=512)
        self.assertTrue(slide.closed)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["valid_patch_count"], 0)
        self.assertTrue(result["feature_extraction_ready"])
        self.assertTrue(all(item["tissue_percentage"] > 20 for item in result["valid_patches"]))

    def test_color_filter_empty_has_explicit_reason(self):
        slide = SyntheticSlide()
        with mock.patch("lung_wsi_mil.preprocessing.pipeline.open_slide", return_value=slide):
            result = preprocess_slide("synthetic.svs", color_distance_threshold=0)
        self.assertEqual(result["reason"], "no_tissue_after_color_filter")
        self.assertEqual(result["valid_patch_count"], 0)

    def test_parameters_reject_wrong_types_nan_and_infinity(self):
        invalid = (
            {"target_magnification": float("nan")},
            {"target_magnification": float("inf")},
            {"patch_size": 512.5},
            {"patch_size": True},
            {"min_tissue_percentage": float("nan")},
            {"color_distance_threshold": float("inf")},
            {"thumbnail_max_size": 1024.5},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                preprocess_slide("not-opened.svs", **kwargs)

    def test_slide_is_closed_when_processing_raises(self):
        slide = SyntheticSlide()
        with (
            mock.patch("lung_wsi_mil.preprocessing.pipeline.open_slide", return_value=slide),
            mock.patch(
                "lung_wsi_mil.preprocessing.pipeline.create_thumbnail",
                side_effect=RuntimeError("boom"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                preprocess_slide("synthetic.svs")
        self.assertTrue(slide.closed)


class PatchAndCoordinateTests(unittest.TestCase):
    def test_fractional_downsample_coordinate_mapping_is_deterministic(self):
        tissue_coordinates = np.array([[0, 0], [100, 100], [511, 511]], dtype=np.int64)
        tiles = get_patch_coordinates(tissue_coordinates, 8.0, 8.0, 3.5, patch_size=512)
        level_zero = tile_indices_to_level0_coordinates(tiles, 3.5, patch_size=512)
        self.assertEqual(level_zero.dtype, np.int64)
        self.assertEqual(len(level_zero), len(np.unique(level_zero, axis=0)))

    def test_tissue_threshold_is_strict_and_extraction_returns_rgb(self):
        self.assertFalse(is_valid_patch(20.0, 20.0))
        self.assertTrue(is_valid_patch(20.0001, 20.0))
        patch = extract_patch(SyntheticSlide(), 0, 0, target_downsample=4.0)
        self.assertEqual(patch.mode, "RGB")
        self.assertEqual(patch.size, (512, 512))
        self.assertTrue(math.isfinite(calculate_patch_tissue_percentage(patch)))


class BatchArtifactTests(unittest.TestCase):
    @staticmethod
    def _result():
        return {
            "valid_patches": [{"x": 1, "y": 2, "tissue_percentage": 50.0}],
            "valid_patch_count": 1,
            "candidate_tile_count": 1,
            "feature_extraction_ready": True,
            "resolution": {
                "source_magnification": 20.0,
                "target_magnification": 5.0,
                "target_mpp_x": 2.0,
                "target_mpp_y": 2.0,
            },
            "thumbnail": Image.new("RGB", (8, 8), "white"),
            "filtered_mask": np.ones((8, 8), dtype=bool),
            "scale_x": 1.0,
            "scale_y": 1.0,
        }

    def test_output_directory_preserves_extension_to_avoid_collision(self):
        root = Path("/input")
        first = slide_output_dir("/output", root, root / "case.svs")
        second = slide_output_dir("/output", root, root / "case.tif")
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "case.svs")

    def test_resume_validates_artifacts_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = root / "slide.svs"
            slide.write_bytes(b"source")
            output = root / "output"
            config = build_config(5, 512, 20, 35, 1024, save_qc=False)
            save_slide_outputs(self._result(), slide, output, config, 1.0, save_qc=False)
            self.assertIsNotNone(load_completed_result(output, config, slide_path=slide))
            coordinates = np.load(output / "coordinates.npy")
            coordinates[0, 0] += 1
            np.save(output / "coordinates.npy", coordinates)
            self.assertIsNone(load_completed_result(output, config, slide_path=slide))
            save_slide_outputs(self._result(), slide, output, config, 1.0, save_qc=False)
            (output / "valid_patches.csv").write_text(
                "x,y,tissue_percentage\n1,2,40.0\n", encoding="utf-8"
            )
            self.assertIsNone(load_completed_result(output, config, slide_path=slide))
            save_slide_outputs(self._result(), slide, output, config, 1.0, save_qc=False)
            (output / "coordinates.npy").unlink()
            self.assertIsNone(load_completed_result(output, config, slide_path=slide))
            save_slide_outputs(self._result(), slide, output, config, 1.0, save_qc=False)
            slide.write_bytes(b"changed source")
            self.assertIsNone(load_completed_result(output, config, slide_path=slide))

    def test_failure_invalidates_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slide = root / "slide.svs"
            slide.write_bytes(b"source")
            output = root / "output"
            config = build_config(5, 512, 20, 35, 1024, save_qc=False)
            save_slide_outputs(self._result(), slide, output, config, 1.0, save_qc=False)
            save_slide_error(slide, output, config, 1.0, RuntimeError("boom"))
            self.assertFalse((output / "result.json").exists())
            self.assertTrue((output / "error.json").is_file())
            self.assertIsNone(load_completed_result(output, config, slide_path=slide))

    def test_output_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = _acquire_output_lock(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "Another preprocessing run"):
                    _acquire_output_lock(root)
            finally:
                lock.unlink()


if __name__ == "__main__":
    unittest.main()
