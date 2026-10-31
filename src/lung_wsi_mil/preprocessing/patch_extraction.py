"""Patch extraction utilities."""

import numpy as np
from PIL import Image
from skimage.morphology import dilation


def calculate_average_background_color(
    patch,
    white_threshold=220,
):
    """Estimate a bright RGB background color for padded slide edges."""
    rgba_array = np.asarray(patch.convert("RGBA"))

    rgb_array = rgba_array[:, :, :3]
    alpha_channel = rgba_array[:, :, 3]

    opaque_mask = alpha_channel > 0
    bright_mask = np.all(rgb_array >= white_threshold, axis=2)
    background_mask = opaque_mask & bright_mask

    background_pixels = rgb_array[background_mask]

    if background_pixels.size == 0:
        return (255, 255, 255)

    average_color = np.rint(background_pixels.mean(axis=0)).astype(np.uint8)

    return tuple(int(value) for value in average_color)


def rgba_to_rgb(patch):
    """Composite OpenSlide RGBA output onto a bright tissue-slide background."""
    if patch.mode != "RGBA":
        return patch.convert("RGB")

    background_color = calculate_average_background_color(patch)
    background = Image.new("RGB", patch.size, background_color)
    background.paste(patch, mask=patch.getchannel("A"))
    return background


def get_level_read_size(
    patch_size,
    target_downsample,
    level_downsample,
):
    """Pixels to read at a chosen pyramid level before exact-scale resizing."""
    read_size = int(round(patch_size * target_downsample / level_downsample))

    return max(read_size, 1)


def extract_patch(
    slide,
    level0_x,
    level0_y,
    target_downsample,
    patch_size=512,
):
    """
    Extract one patch at the requested effective magnification.

    OpenSlide may not contain a native level at the exact target scale. We read
    from the best available pyramid level and resize to patch_size, preserving
    the requested physical field of view.
    """
    level = slide.get_best_level_for_downsample(target_downsample)
    level_downsample = float(slide.level_downsamples[level])

    read_size = get_level_read_size(
        patch_size=patch_size,
        target_downsample=target_downsample,
        level_downsample=level_downsample,
    )

    patch = slide.read_region(
        location=(int(level0_x), int(level0_y)),
        level=level,
        size=(read_size, read_size),
    )

    patch = rgba_to_rgb(patch)

    if patch.size != (patch_size, patch_size):
        patch = patch.resize(
            (patch_size, patch_size),
            Image.Resampling.BILINEAR,
        )

    return patch


def calculate_patch_tissue_percentage(
    patch,
    white_threshold=220,
):
    """Estimate tissue percentage using the reference white-background cutoff."""
    patch_array = np.asarray(patch)

    white_mask = np.all(
        patch_array >= white_threshold,
        axis=2,
    )

    footprint = np.ones((2, 2), dtype=bool)
    white_mask = dilation(white_mask, footprint=footprint)

    tissue_mask = ~white_mask
    return float(tissue_mask.mean() * 100)


def is_valid_patch(
    tissue_percentage,
    min_tissue_percentage=20.0,
):
    """Reference rule: valid patches contain strictly more than 20% tissue."""
    return tissue_percentage > min_tissue_percentage


def filter_valid_patch_coordinates(
    slide,
    level0_coordinates,
    target_downsample,
    patch_size=512,
    min_tissue_percentage=20.0,
):
    """
    Read candidate regions and return coordinates passing the tissue threshold.

    There is intentionally NO minimum number of patches per WSI here. A slide
    with 1, 10, or 29 valid patches is preserved for downstream processing.
    """
    valid_patches = []

    for level0_x, level0_y in level0_coordinates:
        patch = extract_patch(
            slide=slide,
            level0_x=level0_x,
            level0_y=level0_y,
            target_downsample=target_downsample,
            patch_size=patch_size,
        )

        tissue_percentage = calculate_patch_tissue_percentage(patch)

        if is_valid_patch(
            tissue_percentage,
            min_tissue_percentage,
        ):
            valid_patches.append(
                {
                    "x": int(level0_x),
                    "y": int(level0_y),
                    "tissue_percentage": float(tissue_percentage),
                }
            )

    return valid_patches
