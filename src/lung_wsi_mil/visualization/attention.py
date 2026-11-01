"""Patch-aligned attention maps in Level-0 coordinates, with explicit coverage."""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

from lung_wsi_mil.experiments.artifacts import atomic_csv, atomic_json, hash_file, read_json
from lung_wsi_mil.visualization.figures import save_figure


def _validate_attention(coordinates, weights):
    coordinates, weights = np.asarray(coordinates), np.asarray(weights, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not len(coordinates)
        or not np.issubdtype(coordinates.dtype, np.integer)
        or (coordinates < 0).any()
    ):
        raise ValueError("Attention coordinates must be nonempty nonnegative integer [N,2].")
    if (
        weights.shape != (len(coordinates),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1, atol=1e-5)
    ):
        raise ValueError("One finite nonnegative attention weight per patch must sum to one.")
    return coordinates, weights


def percentile_scores(weights):
    """Average-rank percentiles; constant attention is displayed at the midpoint."""
    _, inverse, counts = np.unique(weights, return_inverse=True, return_counts=True)
    if len(weights) == 1:
        return np.full(1, 0.5)
    ranks = np.cumsum(counts) - (counts + 1) / 2
    return ranks[inverse] / (len(weights) - 1)


def rasterize_attention(coordinates, weights, *, slide_dimensions, patch_size_level0, size):
    """Area-rasterize patch rectangles with overlap averaging and no smoothing.

    Unobserved pixels are NaN in the numeric map and transparent in the image.
    Dimensions and patch footprint must come from source geometry, never from
    the maximum sampled coordinate. ``size`` is output (width, height).
    """
    coordinates, weights = _validate_attention(coordinates, weights)
    if (
        len(slide_dimensions) != 2
        or len(size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in (*slide_dimensions, *size)
        )
    ):
        raise ValueError("Slide/output dimensions must be positive integer width/height pairs.")
    if (
        isinstance(patch_size_level0, bool)
        or not math.isfinite(patch_size_level0)
        or patch_size_level0 <= 0
    ):
        raise ValueError("Level-0 patch footprint must be finite and positive.")
    width, height = map(int, slide_dimensions)
    output_width, output_height = map(int, size)
    if (coordinates[:, 0] >= width).any() or (coordinates[:, 1] >= height).any():
        raise ValueError("A patch origin lies outside the full-slide geometry.")
    scale = np.asarray([output_width / width, output_height / height])
    start = np.floor(coordinates * scale).astype(np.int64)
    stop = np.ceil((coordinates + patch_size_level0) * scale).astype(np.int64)
    stop = np.minimum(np.maximum(stop, start + 1), [output_width, output_height])
    x0, y0, x1, y1 = start[:, 0], start[:, 1], stop[:, 0], stop[:, 1]
    coverage = np.zeros((output_height + 1, output_width + 1), dtype=np.int64)
    for xs, ys, sign in ((x0, y0, 1), (x1, y0, -1), (x0, y1, -1), (x1, y1, 1)):
        np.add.at(coverage, (ys, xs), sign)
    coverage = coverage.cumsum(0).cumsum(1)[:-1, :-1]

    def raster(values):
        differences = np.zeros((output_height + 1, output_width + 1), dtype=np.float64)
        for xs, ys, sign in ((x0, y0, 1), (x1, y0, -1), (x0, y1, -1), (x1, y1, 1)):
            np.add.at(differences, (ys, xs), sign * values)
        totals = differences.cumsum(0).cumsum(1)[:-1, :-1]
        return np.divide(totals, coverage, out=np.full_like(totals, np.nan), where=coverage > 0)

    return raster(weights), raster(percentile_scores(weights)), coverage


def export_attention(directory, record, coordinates, weights, provenance):
    coordinates, weights = _validate_attention(coordinates, weights)
    directory = Path(directory)
    if any(character in record["slide_id"] for character in "/\\") or record["slide_id"] in {
        ".",
        "..",
    }:
        raise ValueError("slide_id must be a safe single path component.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['slide_id']}.csv"
    atomic_csv(
        path,
        [
            {
                "slide_id": record["slide_id"],
                "patient_id": record["patient_id"],
                "view": "Original",
                "patch_index": index,
                "x_level0": int(x),
                "y_level0": int(y),
                "attention": float(weight),
            }
            for index, ((x, y), weight) in enumerate(zip(coordinates, weights, strict=True))
        ],
    )
    atomic_json(
        path.with_suffix(".json"),
        {
            "schema_version": 1,
            "prediction": record,
            "provenance": provenance,
            "attention_csv_sha256": hash_file(path),
            "attention_sum": float(weights.sum()),
            "patches": len(weights),
            "interpretation": "Within-slide model attention, not tumor segmentation or a clinical diagnosis.",
        },
    )
    return path


def read_attention(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (
        not rows
        or len({row["slide_id"] for row in rows}) != 1
        or any(row["view"] != "Original" for row in rows)
    ):
        raise ValueError("One Original-only slide is required in an attention CSV.")
    coordinates = np.asarray(
        [[int(row["x_level0"]), int(row["y_level0"])] for row in rows], dtype=np.int64
    )
    weights = np.asarray([float(row["attention"]) for row in rows])
    _validate_attention(coordinates, weights)
    metadata_path = Path(path).with_suffix(".json")
    if metadata_path.is_file() and read_json(metadata_path).get(
        "attention_csv_sha256"
    ) != hash_file(path):
        raise ValueError("Attention CSV differs from its exported provenance.")
    return rows[0]["slide_id"], coordinates, weights


def _save_numpy(path, values):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_from_preprocessing(
    attention_csv, preprocessing, output, *, max_size=1024, alpha=0.45, title=None
):
    """Use the saved preprocessing geometry and original QC thumbnail when present."""
    if isinstance(max_size, bool) or not isinstance(max_size, int) or not 1 <= max_size <= 4096:
        raise ValueError("max_size must be an integer in [1,4096].")
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1].")
    attention_csv, preprocessing, output = map(Path, (attention_csv, preprocessing, output))
    slide, coordinates, weights = read_attention(attention_csv)
    payload = read_json(preprocessing / "result.json")
    if payload.get("status") != "success":
        raise ValueError("A successful preprocessing result is required for spatial rendering.")
    result = payload["result"]
    dimensions = tuple(result["slide_info"]["dimensions"])
    if len(dimensions) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions
    ):
        raise ValueError("Preprocessing slide dimensions must be positive integer width/height.")
    footprint = result["resolution"]["level0_patch_size"]
    source_coordinates = np.load(preprocessing / "coordinates.npy", allow_pickle=False)
    if (
        source_coordinates.dtype != np.dtype("int64")
        or not np.array_equal(source_coordinates, coordinates)
        or result["valid_patch_count"] != len(weights)
    ):
        raise ValueError("Attention coordinates/order differ from the preprocessing source.")
    scale = min(1, max_size / max(dimensions))
    size = tuple(max(1, int(round(value * scale))) for value in dimensions)
    thumbnail, thumbnail_path = None, None
    qc = payload.get("files", {}).get("qc")
    if qc and qc.get("thumbnail"):
        thumbnail_path = (preprocessing / qc["thumbnail"]).resolve()
        if (
            not thumbnail_path.is_relative_to(preprocessing.resolve())
            or not thumbnail_path.is_file()
        ):
            raise ValueError(
                "Recorded preprocessing thumbnail is unavailable or outside its source folder."
            )
        with Image.open(thumbnail_path) as image:
            expected_ratio = dimensions[0] / dimensions[1]
            if abs(image.width - image.height * expected_ratio) > max(2, image.width * 0.01):
                raise ValueError("Thumbnail aspect ratio conflicts with full-slide geometry.")
            thumbnail = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    raw, display, coverage = rasterize_attention(
        coordinates, weights, slide_dimensions=dimensions, patch_size_level0=footprint, size=size
    )
    output.mkdir(parents=True, exist_ok=True)
    _save_numpy(output / "attention_raw.npy", raw.astype(np.float32))
    _save_numpy(output / "coverage.npy", coverage.astype(np.uint32))
    rgba = (colormaps["viridis"](np.nan_to_num(display, nan=0.5)) * 255).round().astype(np.uint8)
    rgba[:, :, 3] = np.where(coverage > 0, 255, 0)
    Image.fromarray(rgba).save(output / "heatmap.png")
    Image.fromarray((coverage > 0).astype(np.uint8) * 255).save(output / "coverage.png")
    overlay = None
    if thumbnail is not None:
        layer = rgba.copy()
        layer[:, :, 3] = np.where(coverage > 0, int(round(alpha * 255)), 0)
        overlay = Image.alpha_composite(thumbnail.convert("RGBA"), Image.fromarray(layer)).convert(
            "RGB"
        )
        overlay.save(output / "overlay.png")
    figure = Figure(figsize=(12 if thumbnail else 8, 4.1), layout="constrained")
    axes = np.atleast_1d(figure.subplots(1, 3 if thumbnail else 2))
    extent = (0, dimensions[0], dimensions[1], 0)
    if thumbnail:
        axes[0].imshow(thumbnail, extent=extent)
        axes[0].set_title("Source thumbnail")
        axes[1].imshow(rgba, extent=extent)
        axes[1].set_title("Attention · observed patches only")
        axes[2].imshow(overlay, extent=extent)
        axes[2].set_title("Patch-aligned overlay")
        color_axis = axes[1]
    else:
        axes[0].imshow(rgba, extent=extent)
        axes[0].set_title("Attention · no image supplied")
        axes[1].imshow(coverage > 0, cmap="Greys", extent=extent, vmin=0, vmax=1)
        axes[1].set_title("Observed patch coverage")
        color_axis = axes[0]
    for axis in axes:
        axis.set(xlabel="Level-0 x (pixels)", ylabel="Level-0 y (pixels)")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        ax=color_axis,
        fraction=0.05,
        label="Within-slide attention percentile",
    )
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    figure.suptitle(title or slide, fontsize=11)
    save_figure(figure, output / "overview", formats=("png",))
    metadata = {
        "slide_id": slide,
        "slide_dimensions_level0": list(dimensions),
        "patch_size_level0": float(footprint),
        "raster_size": list(size),
        "patches": len(weights),
        "covered_pixels": int((coverage > 0).sum()),
        "display": "within-slide average-rank percentiles; overlapping footprints averaged",
        "unsampled_regions": "NaN in raw numeric map, transparent in heatmap; no extrapolation or smoothing",
        "overlay_generated": overlay is not None,
        "alpha": alpha,
        "source_attention_sha256": hash_file(attention_csv),
        "source_preprocessing_sha256": hash_file(preprocessing / "result.json"),
        "source_coordinates_sha256": hash_file(preprocessing / "coordinates.npy"),
        "source_thumbnail_sha256": hash_file(thumbnail_path) if thumbnail_path else None,
        "interpretation": "Model attention is not a tumor mask or a localization ground truth.",
    }
    atomic_json(output / "heatmap.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new spatial-map output folder; existing maps are preserved.")
    if (
        args.output.resolve().is_relative_to(args.attention.resolve().parent)
        or args.attention.resolve().is_relative_to(args.output.resolve())
        or args.output.resolve().is_relative_to(args.preprocessing.resolve())
        or args.preprocessing.resolve().is_relative_to(args.output.resolve())
    ):
        parser.error("Choose a separate output folder outside the attention source folder.")
    try:
        render_from_preprocessing(
            args.attention, args.preprocessing, args.output, max_size=args.max_size
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Spatial attention map saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
