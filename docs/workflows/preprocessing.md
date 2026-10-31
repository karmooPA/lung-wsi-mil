# Lung WSI preprocessing

This folder implements the preprocessing stage used before KimiaNet feature extraction.

Default settings reproduce the current project target:

- Target magnification: **5x**
- Patch size: **512 x 512**
- Color-distance threshold: **35**
- Minimum tissue percentage: **strictly > 20%**
- **No minimum patch-count cutoff per WSI**

## Important behavior

A WSI with 1, 10, or 29 valid patches is preserved. There is no 30-patch rejection rule in this preprocessing implementation.

The pipeline does not save thousands of patch PNG files. It stores Level-0 coordinates and tissue percentages so the feature-extraction stage can read patches directly from the WSI.

## Single-slide API

```python
from lung_wsi_mil.preprocessing.pipeline import preprocess_slide

result = preprocess_slide("/path/to/slide.svs")
print(result["valid_patch_count"])
print(result["valid_patches"][:5])
```

## Full-dataset runner

From the repository root:

```bash
python -m lung_wsi_mil.preprocessing.batch_preprocessing \
  "/path/to/TCGA_LUSC" \
  "/path/to/preprocessing_output"
```

For a small QC run, add:

```bash
--save-qc
```

The runner recursively discovers WSI files, continues after corrupted/unsupported slides, writes an incremental atomic `manifest.csv`, and resumes only when configuration, source identity, coordinate artifacts and requested QC files remain valid.

Use `--force` to reprocess completed slides.

## Per-slide outputs

Each WSI gets its own mirrored output directory containing:

- `result.json`: metadata, resolved magnification, counts, valid patch list, parameters
- `coordinates.npy`: Level-0 `(x, y)` coordinates for feature extraction
- `valid_patches.csv`: coordinates + tissue percentage
- `qc/` (only with `--save-qc`): thumbnail, filtered mask, valid-patch overlay
- `error.json`: only if processing failed

The root output directory also contains `manifest.csv`.

## Magnification handling

Patch coordinates are defined in Level-0 space. Objective power is used when it agrees with the MPP-derived magnification. If objective power is missing, source magnification is estimated from MPP using the reference-project convention `magnification ~= 10 / MPP`. The pipeline fails safely instead of guessing when objective and MPP disagree by more than 20% or MPP X/Y anisotropy exceeds 10%.

OpenSlide's best pyramid level is used only for efficient reading; patches are resized so the requested effective magnification is preserved even when the WSI has no native 5x level.

## Morphology note

The reference project performs erosion followed by dilation on an OpenCV binary image where tissue is black and background is white. This code stores tissue as `True`, so the polarity-equivalent operation on the tissue foreground is `closing` with a 3x3 footprint.
