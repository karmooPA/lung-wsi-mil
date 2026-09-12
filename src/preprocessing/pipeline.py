import numpy as np

from src.preprocessing.wsi_reader import (
    open_slide,
    get_slide_info,
    resolve_target_resolution,
)
from src.preprocessing.thumbnail import create_thumbnail
from src.preprocessing.tissue_detection import (
    image_to_array,
    rgb_to_gray,
    create_tissue_mask,
    clean_tissue_mask,
    get_average_tissue_color,
    calculate_color_distance,
    filter_by_color_distance,
)
from src.preprocessing.patch_coordinates import (
    get_tissue_coordinates,
    get_patch_coordinates,
    tile_indices_to_level0_coordinates,
)
from src.preprocessing.patch_extraction import filter_valid_patch_coordinates


def _validate_parameters(
    target_magnification,
    patch_size,
    min_tissue_percentage,
    color_distance_threshold,
    thumbnail_max_size,
):
    if target_magnification <= 0:
        raise ValueError("target_magnification must be greater than 0.")
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than 0.")
    if not 0 <= min_tissue_percentage <= 100:
        raise ValueError("min_tissue_percentage must be between 0 and 100.")
    if color_distance_threshold < 0:
        raise ValueError("color_distance_threshold cannot be negative.")
    if thumbnail_max_size <= 0:
        raise ValueError("thumbnail_max_size must be greater than 0.")


def _empty_result(
    slide_info,
    resolution,
    thumbnail,
    scale_x,
    scale_y,
    otsu_threshold,
    cleaned_mask,
    parameters,
    reason,
):
    return {
        "status": "success",
        "reason": reason,
        "parameters": parameters,
        "slide_info": slide_info,
        "resolution": resolution,
        "thumbnail": thumbnail,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "otsu_threshold": float(otsu_threshold),
        "average_tissue_color": None,
        "filtered_mask": np.zeros_like(cleaned_mask, dtype=bool),
        "candidate_tile_count": 0,
        "valid_patch_count": 0,
        "valid_patches": [],
        "feature_extraction_ready": False,
    }


def preprocess_slide(
    slide_path,
    target_magnification=5,
    patch_size=512,
    min_tissue_percentage=20.0,
    color_distance_threshold=35,
    thumbnail_max_size=1024,
):
    """
    Run WSI preprocessing and return valid Level-0 patch coordinates.

    No slide is rejected because it has fewer than an arbitrary number of
    patches. Any positive number of valid patches is considered ready for the
    feature-extraction stage.
    """
    _validate_parameters(
        target_magnification=target_magnification,
        patch_size=patch_size,
        min_tissue_percentage=min_tissue_percentage,
        color_distance_threshold=color_distance_threshold,
        thumbnail_max_size=thumbnail_max_size,
    )

    parameters = {
        "target_magnification": float(target_magnification),
        "patch_size": int(patch_size),
        "min_tissue_percentage": float(min_tissue_percentage),
        "color_distance_threshold": float(color_distance_threshold),
        "thumbnail_max_size": int(thumbnail_max_size),
    }

    slide = open_slide(slide_path)

    try:
        slide_info = get_slide_info(slide)
        resolution = resolve_target_resolution(
            slide=slide,
            slide_info=slide_info,
            target_magnification=target_magnification,
        )

        resolution["level0_patch_size"] = float(
            patch_size * resolution["target_downsample"]
        )

        thumbnail, scale_x, scale_y = create_thumbnail(
            slide,
            max_size=thumbnail_max_size,
        )

        image_array = image_to_array(thumbnail)
        gray_image = rgb_to_gray(image_array)

        tissue_mask, otsu_threshold = create_tissue_mask(gray_image)
        cleaned_mask = clean_tissue_mask(tissue_mask)

        if not np.any(cleaned_mask):
            return _empty_result(
                slide_info=slide_info,
                resolution=resolution,
                thumbnail=thumbnail,
                scale_x=scale_x,
                scale_y=scale_y,
                otsu_threshold=otsu_threshold,
                cleaned_mask=cleaned_mask,
                parameters=parameters,
                reason="no_tissue_after_morphology",
            )

        average_color = get_average_tissue_color(
            image_array,
            cleaned_mask,
        )

        color_distance = calculate_color_distance(
            image_array,
            average_color,
        )

        filtered_mask = filter_by_color_distance(
            cleaned_mask,
            color_distance,
            threshold=color_distance_threshold,
        )

        tissue_coordinates = get_tissue_coordinates(filtered_mask)

        tile_coordinates = get_patch_coordinates(
            tissue_coordinates=tissue_coordinates,
            scale_x=scale_x,
            scale_y=scale_y,
            target_downsample=resolution["target_downsample"],
            patch_size=patch_size,
        )

        level0_coordinates = tile_indices_to_level0_coordinates(
            tile_coordinates=tile_coordinates,
            target_downsample=resolution["target_downsample"],
            patch_size=patch_size,
        )

        valid_patches = filter_valid_patch_coordinates(
            slide=slide,
            level0_coordinates=level0_coordinates,
            target_downsample=resolution["target_downsample"],
            patch_size=patch_size,
            min_tissue_percentage=min_tissue_percentage,
        )

        return {
            "status": "success",
            "reason": None,
            "parameters": parameters,
            "slide_info": slide_info,
            "resolution": resolution,
            "thumbnail": thumbnail,
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "otsu_threshold": float(otsu_threshold),
            "average_tissue_color": average_color,
            "filtered_mask": filtered_mask,
            "candidate_tile_count": int(len(tile_coordinates)),
            "valid_patch_count": int(len(valid_patches)),
            "valid_patches": valid_patches,
            "feature_extraction_ready": len(valid_patches) > 0,
        }

    finally:
        slide.close()
