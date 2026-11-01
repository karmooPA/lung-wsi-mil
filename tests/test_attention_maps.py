"""Spatial arithmetic independent of model predictions and image appearance."""

import numpy as np
import pytest

from lung_wsi_mil.visualization.attention import (
    export_attention,
    rasterize_attention,
    render_from_preprocessing,
)
from tests.experiment_fixtures import make_experiment_inputs


def test_overlap_averages_weights_and_leaves_unobserved_pixels_unknown():
    coordinates = np.asarray([[0, 0], [2, 0]], dtype=np.int64)
    raw, display, coverage = rasterize_attention(
        coordinates, [0.2, 0.8], slide_dimensions=(8, 4), patch_size_level0=4, size=(8, 4)
    )
    assert raw[0, 0] == pytest.approx(0.2)
    assert raw[0, 2] == pytest.approx(0.5)
    assert raw[0, 5] == pytest.approx(0.8)
    assert coverage[0, 2] == 2
    assert np.isnan(raw[0, 7]) and coverage[0, 7] == 0
    assert display[0, 2] == pytest.approx(0.5)


def test_patch_geometry_is_scaled_independently_on_both_axes_and_edges_clipped():
    raw, _, coverage = rasterize_attention(
        np.asarray([[6, 4]], dtype=np.int64),
        [1.0],
        slide_dimensions=(8, 6),
        patch_size_level0=4,
        size=(4, 3),
    )
    assert coverage.sum() == 1
    assert raw[2, 3] == 1
    assert np.isnan(raw[0, 0])


def test_constant_attention_is_a_neutral_percentile_not_false_contrast():
    _, display, _ = rasterize_attention(
        np.asarray([[0, 0], [2, 0]], dtype=np.int64),
        [0.5, 0.5],
        slide_dimensions=(4, 2),
        patch_size_level0=2,
        size=(4, 2),
    )
    assert np.all(display == 0.5)


def test_spatial_render_uses_real_source_geometry_and_rejects_misalignment(tmp_path):
    source = make_experiment_inputs(tmp_path / "source")
    slide = "slide-000-0"
    pre = source["patch_root"] / "slides" / slide / "preprocessing"
    coords = np.load(pre / "coordinates.npy", allow_pickle=False)
    record = {
        "slide_id": slide,
        "patient_id": "patient-000",
        "y_true": 0,
        "probability_tumor": 0.2,
        "y_pred": 0,
        "threshold": 0.5,
        "view": "Original",
    }
    table = export_attention(
        tmp_path / "attention", record, coords, np.full(len(coords), 1 / len(coords)), {}
    )
    result = render_from_preprocessing(table, pre, tmp_path / "map", max_size=128)
    assert result["overlay_generated"]
    assert result["raster_size"] == [128, 64]
    assert (tmp_path / "map/overlay.png").is_file()
    np.save(pre / "coordinates.npy", coords[::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="coordinates/order"):
        render_from_preprocessing(table, pre, tmp_path / "bad-map", max_size=128)
