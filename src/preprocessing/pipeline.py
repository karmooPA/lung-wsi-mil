from src.preprocessing.wsi_reader import (
    open_slide,
    get_slide_info,
)

from src.preprocessing.thumbnail import (
    create_thumbnail,
)

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

from src.preprocessing.patch_extraction import (
    filter_valid_patch_coordinates,
)


def preprocess_slide(
    slide_path,
    target_magnification=5,
    patch_size=512,
    min_tissue_percentage=20.0,
    color_distance_threshold=35,
):
    slide = open_slide(slide_path)

    try:
        # 1. Read WSI metadata
        slide_info = get_slide_info(slide)

        objective_power = slide_info["objective_power"]

        if objective_power is None:
            mpp = slide_info["mpp_x"]

            if mpp is None:
                mpp = slide_info["mpp_y"]

            if mpp is None or mpp <= 0:
                raise ValueError(
                    "Objective power and MPP are missing from slide metadata."
                )

            objective_power = 10.0 / mpp

        # 2. Generate thumbnail
        thumbnail, scale_x, scale_y = create_thumbnail(slide)

        # 3. Convert thumbnail to NumPy array
        image_array = image_to_array(thumbnail)

        # 4. Convert RGB thumbnail to grayscale
        gray_image = rgb_to_gray(image_array)

        # 5. Otsu tissue detection
        tissue_mask, otsu_threshold = create_tissue_mask(
            gray_image
        )

        # 6. Morphological cleaning
        cleaned_mask = clean_tissue_mask(
            tissue_mask
        )

        # 7. Calculate average tissue color
        average_color = get_average_tissue_color(
            image_array,
            cleaned_mask,
        )

        # 8. Calculate RGB color distance
        color_distance = calculate_color_distance(
            image_array,
            average_color,
        )

        # 9. Apply color-distance filtering
        filtered_mask = filter_by_color_distance(
            cleaned_mask,
            color_distance,
            threshold=color_distance_threshold,
        )

        # 10. Get tissue-pixel coordinates on thumbnail
        tissue_coordinates = get_tissue_coordinates(
            filtered_mask
        )

        # 11. Convert tissue pixels to unique 5x tile indices
        tile_coordinates = get_patch_coordinates(
            tissue_coordinates=tissue_coordinates,
            scale_x=scale_x,
            scale_y=scale_y,
            objective_power=objective_power,
            target_magnification=target_magnification,
            patch_size=patch_size,
        )

        # 12. Convert tile indices to Level-0 coordinates
        level0_coordinates = tile_indices_to_level0_coordinates(
            tile_coordinates=tile_coordinates,
            objective_power=objective_power,
            target_magnification=target_magnification,
            patch_size=patch_size,
        )

        # 13. Extract candidate patches and keep valid ones
        valid_patches = filter_valid_patch_coordinates(
            slide=slide,
            level0_coordinates=level0_coordinates,
            objective_power=objective_power,
            target_magnification=target_magnification,
            patch_size=patch_size,
            min_tissue_percentage=min_tissue_percentage,
        )

        return {
            "slide_info": slide_info,
            "thumbnail": thumbnail,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "otsu_threshold": otsu_threshold,
            "average_tissue_color": average_color,
            "filtered_mask": filtered_mask,
            "candidate_tile_count": len(tile_coordinates),
            "valid_patch_count": len(valid_patches),
            "valid_patches": valid_patches,
        }

    finally:
        slide.close()