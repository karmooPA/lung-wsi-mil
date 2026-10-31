"""Patch coordinates utilities."""

import numpy as np


def get_tissue_coordinates(filtered_mask):
    """Return thumbnail tissue coordinates as (x, y) pairs."""
    y_coords, x_coords = np.where(filtered_mask)
    return np.column_stack((x_coords, y_coords))


def get_patch_coordinates(
    tissue_coordinates,
    scale_x,
    scale_y,
    target_downsample,
    patch_size=512,
):
    """
    Convert thumbnail tissue pixels to unique tile indices at target scale.
    """
    if tissue_coordinates.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    level0_x = tissue_coordinates[:, 0] * scale_x
    level0_y = tissue_coordinates[:, 1] * scale_y

    target_x = level0_x / target_downsample
    target_y = level0_y / target_downsample

    tile_x = np.floor(target_x / patch_size).astype(np.int64)
    tile_y = np.floor(target_y / patch_size).astype(np.int64)

    tile_coordinates = np.column_stack((tile_x, tile_y))
    return np.unique(tile_coordinates, axis=0)


def tile_indices_to_level0_coordinates(
    tile_coordinates,
    target_downsample,
    patch_size=512,
):
    """Convert target-scale tile indices to OpenSlide Level-0 coordinates."""
    if tile_coordinates.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    level0_coordinates = tile_coordinates * patch_size * target_downsample
    return np.rint(level0_coordinates).astype(np.int64)
