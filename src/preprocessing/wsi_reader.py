import openslide


def open_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide


def to_float(value):
    if value is None:
        return None

    return float(value)


def get_slide_info(slide):
    info = {
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
    }

    return info