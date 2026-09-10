def create_thumbnail(slide, max_size=1024):
    thumbnail = slide.get_thumbnail((max_size, max_size))
    thumbnail = thumbnail.convert("RGB")

    scale_x = slide.dimensions[0] / thumbnail.size[0]
    scale_y = slide.dimensions[1] / thumbnail.size[1]

    return thumbnail, scale_x, scale_y

