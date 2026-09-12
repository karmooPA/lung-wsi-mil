from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np


def create_patch_overlay(result, outline="red", width=2):
    """Draw valid patch footprints on the WSI thumbnail."""
    overlay = result["thumbnail"].copy()
    draw = ImageDraw.Draw(overlay)

    scale_x = result["scale_x"]
    scale_y = result["scale_y"]
    level0_patch_size = result["resolution"]["level0_patch_size"]

    for patch in result["valid_patches"]:
        x0 = int(round(patch["x"] / scale_x))
        y0 = int(round(patch["y"] / scale_y))
        x1 = int(round((patch["x"] + level0_patch_size) / scale_x))
        y1 = int(round((patch["y"] + level0_patch_size) / scale_y))

        draw.rectangle(
            [x0, y0, x1, y1],
            outline=outline,
            width=width,
        )

    return overlay


def save_qc_images(result, output_dir):
    """Save thumbnail, filtered mask, and valid-patch overlay."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbnail_path = output_dir / "thumbnail.png"
    mask_path = output_dir / "filtered_mask.png"
    overlay_path = output_dir / "valid_patches_overlay.png"

    result["thumbnail"].save(thumbnail_path)

    mask_image = Image.fromarray(
        result["filtered_mask"].astype(np.uint8) * 255
    )
    mask_image.save(mask_path)

    overlay = create_patch_overlay(result)
    overlay.save(overlay_path)

    return {
        "thumbnail": str(thumbnail_path),
        "filtered_mask": str(mask_path),
        "valid_patches_overlay": str(overlay_path),
    }
