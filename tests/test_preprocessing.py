from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

from src.preprocessing.pipeline import preprocess_slide


slide_path = "/Users/mohammad/Desktop/TCGA_LUSC_TWO_SAMPLES/Tumor_label_1/TCGA-68-7756-01A-01-TS1.206097ad-f83b-4203-a770-134f24d8f398.svs"


# Folder for test/QC outputs
output_dir = Path(
    "/Users/mohammad/Desktop/TCGA_LUSC_TEST_OUTPUT/Tumor"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


result = preprocess_slide(
    slide_path=slide_path,
    target_magnification=5,
    patch_size=512,
    min_tissue_percentage=20.0,
    color_distance_threshold=35,
)


print("\n--- SLIDE INFO ---")
print(result["slide_info"])

print("\n--- PREPROCESSING RESULT ---")
print("Otsu threshold:", result["otsu_threshold"])
print("Average tissue color:", result["average_tissue_color"])
print("Candidate tile count:", result["candidate_tile_count"])
print("Valid patch count:", result["valid_patch_count"])

print("\n--- FIRST 5 VALID PATCHES ---")
print(result["valid_patches"][:5])


# Save thumbnail
thumbnail_path = output_dir / "tumor_thumbnail.png"

result["thumbnail"].save(thumbnail_path)


# Save filtered mask
mask_image = Image.fromarray(
    result["filtered_mask"].astype(np.uint8) * 255
)

mask_path = output_dir / "tumor_filtered_mask.png"

mask_image.save(mask_path)


# Draw valid patch boxes on thumbnail
thumbnail_with_boxes = result["thumbnail"].copy()

draw = ImageDraw.Draw(thumbnail_with_boxes)

objective_power = result["slide_info"]["objective_power"]

target_magnification = 5
patch_size = 512

level0_patch_size = int(
    patch_size
    * (objective_power / target_magnification)
)

scale_x = result["scale_x"]
scale_y = result["scale_y"]


for patch in result["valid_patches"]:

    x0 = int(
        patch["x"] / scale_x
    )

    y0 = int(
        patch["y"] / scale_y
    )

    x1 = int(
        (patch["x"] + level0_patch_size)
        / scale_x
    )

    y1 = int(
        (patch["y"] + level0_patch_size)
        / scale_y
    )

    draw.rectangle(
        [x0, y0, x1, y1],
        outline="red",
        width=2,
    )


overlay_path = (
    output_dir
    / "tumor_valid_patches_overlay.png"
)

thumbnail_with_boxes.save(
    overlay_path
)


print("\n--- QC IMAGES ---")
print("Saved:", thumbnail_path)
print("Saved:", mask_path)
print("Saved:", overlay_path)