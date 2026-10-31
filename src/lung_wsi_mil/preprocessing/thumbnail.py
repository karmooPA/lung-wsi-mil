"""Thumbnail utilities."""


def create_thumbnail(slide, max_size=1024):
    """Create an RGB thumbnail and Level-0-to-thumbnail scale factors."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0.")

    thumbnail = slide.get_thumbnail((max_size, max_size))
    thumbnail = thumbnail.convert("RGB")

    scale_x = slide.dimensions[0] / thumbnail.size[0]
    scale_y = slide.dimensions[1] / thumbnail.size[1]

    return thumbnail, scale_x, scale_y
