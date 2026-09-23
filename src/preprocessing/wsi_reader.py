import openslide


REFERENCE_MPP_TO_MAGNIFICATION_CONSTANT = 10.0


def open_slide(slide_path):
    """Open a whole-slide image with OpenSlide."""
    return openslide.OpenSlide(str(slide_path))


def to_float(value):
    """Convert an OpenSlide metadata value to float when possible."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_slide_info(slide):
    """Collect the metadata needed by the preprocessing pipeline."""
    return {
        "dimensions": slide.dimensions,
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": slide.level_downsamples,
        "mpp_x": to_float(
            slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        ),
        "mpp_y": to_float(
            slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        ),
        "objective_power": to_float(
            slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        ),
        "vendor": slide.properties.get(openslide.PROPERTY_NAME_VENDOR),
    }


def _positive(value):
    return value is not None and value > 0


def estimate_objective_power_from_mpp(mpp_x, mpp_y):
    """
    Estimate source magnification from MPP using the convention used by the
    reference WSI_MIL_ROI code: magnification ~= 10 / MPP.
    """
    mpp_values = [mpp for mpp in (mpp_x, mpp_y) if _positive(mpp)]

    if not mpp_values:
        return None

    mean_mpp = sum(mpp_values) / len(mpp_values)
    return REFERENCE_MPP_TO_MAGNIFICATION_CONSTANT / mean_mpp


def resolve_target_resolution(slide, slide_info, target_magnification):
    """
    Resolve the exact physical scale used for patch extraction.

    Coordinates remain in OpenSlide Level-0 space. A suitable pyramid level is
    chosen only for efficient reading, and the extracted region is resized to
    the requested patch size later. This keeps 5x consistent even when the WSI
    has no native 5x pyramid level.
    """
    if target_magnification <= 0:
        raise ValueError("target_magnification must be greater than 0.")

    objective_power = slide_info["objective_power"]
    estimated_power = estimate_objective_power_from_mpp(
        slide_info["mpp_x"],
        slide_info["mpp_y"],
    )

    if _positive(objective_power):
        source_magnification = objective_power
        source = "objective_power"
    elif _positive(estimated_power):
        source_magnification = estimated_power
        source = "mpp_estimate"
    else:
        raise ValueError(
            "Cannot determine source magnification: both objective power and "
            "MPP are missing or invalid."
        )

    if target_magnification > source_magnification:
        raise ValueError(
            f"Target magnification ({target_magnification}x) is higher than "
            f"the source magnification ({source_magnification:.3f}x)."
        )

    target_downsample = source_magnification / target_magnification
    best_level = slide.get_best_level_for_downsample(target_downsample)
    best_level_downsample = float(slide.level_downsamples[best_level])

    warning = None
    if _positive(objective_power) and _positive(estimated_power):
        relative_difference = abs(objective_power - estimated_power) / objective_power
        if relative_difference > 0.20:
            warning = (
                "Objective power and MPP-derived magnification differ by more "
                "than 20%. Objective power was used for extraction scale."
            )

    return {
        "source_magnification": float(source_magnification),
        "source_magnification_method": source,
        "mpp_estimated_source_magnification": (
            float(estimated_power) if estimated_power is not None else None
        ),
        "target_magnification": float(target_magnification),
        "target_downsample": float(target_downsample),
        "best_openslide_level": int(best_level),
        "best_openslide_level_downsample": best_level_downsample,
        "target_mpp_x": (
            float(slide_info["mpp_x"] * target_downsample)
            if _positive(slide_info["mpp_x"])
            else None
        ),
        "target_mpp_y": (
            float(slide_info["mpp_y"] * target_downsample)
            if _positive(slide_info["mpp_y"])
            else None
        ),
        "metadata_warning": warning,
    }
