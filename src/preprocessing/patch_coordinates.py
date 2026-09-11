import numpy as np


def get_tissue_coordinates(filtered_mask):
    y_coords, x_coords = np.where(filtered_mask)

    coordinates = np.column_stack((x_coords, y_coords))

    return coordinates


def get_patch_coordinates(
    tissue_coordinates,
    scale_x,
    scale_y,
    objective_power,
    target_magnification=5,
    patch_size=512,
):
    magnification_downsample = objective_power / target_magnification

    level0_x = tissue_coordinates[:, 0] * scale_x
    level0_y = tissue_coordinates[:, 1] * scale_y

    target_x = level0_x / magnification_downsample
    target_y = level0_y / magnification_downsample

    tile_x = np.floor(target_x / patch_size).astype(int)
    tile_y = np.floor(target_y / patch_size).astype(int)

    tile_coordinates = np.column_stack((tile_x, tile_y))
    tile_coordinates = np.unique(tile_coordinates, axis=0)

    return tile_coordinates


def tile_indices_to_level0_coordinates(
    tile_coordinates,
    objective_power,
    target_magnification=5,
    patch_size=512,
):
    magnification_downsample = objective_power / target_magnification

    level0_coordinates = (
        tile_coordinates
        * patch_size
        * magnification_downsample
    )

    level0_coordinates = np.rint(level0_coordinates).astype(np.int64)

    return level0_coordinates
