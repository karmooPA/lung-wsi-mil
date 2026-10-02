# 14-WSI preprocessing + augmentation pilot

This runner connects the repository's existing WSI preprocessing and data
augmentation stages for a small balanced integration cohort: **7 Tumor + 7
Normal** WSIs.

The accepted real-data run and per-slide hashes are recorded in
[pilot-14-wsi-results.md](pilot-14-wsi-results.md).

It does not change the scientific operators. Preprocessing still produces valid
level-0 coordinates at 5x using 512x512 patches and the existing tissue/color
filters. Each valid patch is then extracted and passed through the existing
`transform_batch`, producing Original, Aug1 and Aug2.

## Run

From the repository root:

```bash
python -m pip install -r requirements-augmentation.txt
python scripts/run_14_wsi_pilot.py \
  /path/to/downloaded/TCGA-LUSC \
  ../lung-wsi-mil-outputs/pilot14-final \
  --inventory metadata/dataset_inventory.csv \
  --seed 42 \
  --verify-source-md5
```

The WSI files may be in the expected GDC directory layout or in another folder
layout. The runner first tries `relative_path`, then the unique inventory
filename, then the TCGA slide-id filename prefix.

By default it selects seven Tumor and seven Normal slides that are actually
present and uses distinct patient IDs across the 14 selected slides. The
selection is deterministic for a given `--seed` and is written to
`../lung-wsi-mil-outputs/pilot14-final/selection.csv` before processing starts.

To pin an exact cohort, provide a CSV with one `slide_id` column and exactly 14
rows (7 Tumor + 7 Normal):

```bash
python scripts/run_14_wsi_pilot.py \
  ../lung-wsi-mil-inputs/pilot14 \
  ../lung-wsi-mil-outputs/pilot14-final \
  --selection-csv metadata/pilot14_selection.csv --verify-source-md5
```

`bash scripts/run_pilot14.sh` runs this exact pinned cohort with the accepted
seed, chunk size and QC coverage. See [workspace-layout.md](workspace-layout.md)
for the three-folder WSL layout. Inputs and outputs must not overlap, and the
pilot runner rejects outputs inside the repository.

## Saved outputs

```text
../lung-wsi-mil-outputs/pilot14-final/
├── run_config.json
├── selection.csv
├── manifest.csv
└── slides/
    └── <slide_id>/
        ├── preprocessing/
        │   ├── coordinates.npy
        │   ├── valid_patches.csv
        │   ├── result.json
        │   └── qc/...
        ├── augmentation/
        │   ├── patch_views.h5
        │   ├── augmentation_quality.csv
        │   ├── augmentation_quality_summary.json
        │   ├── augmentation_qc_patch_0000.png
        │   └── augmentation_qc_montage.png
        └── complete.json
```

`patch_views.h5` contains:

- `images`: uint8 `[3, N, 512, 512, 3]`, ordered Original / Aug1 / Aug2;
- `coords`: int64 `[N, 2]`, the matching level-0 `(x, y)` coordinates;
- `tissue_percentage`: float32 `[N]`.

The HDF5 image representation is a lossless compact encoding of the exact
normalized float32 tensors returned by `transform_batch`. It reduces raw storage
from 9 MiB per patch to 2.25 MiB per patch before HDF5 compression. The helper
below reconstructs the exact `[3, B, 3, 512, 512]` float32 KimiaNet input:

```python
from src.augmentation.storage import load_augmented_chunk

transformed, coords, tissue = load_augmented_chunk(
    "../lung-wsi-mil-outputs/pilot14-final/slides/<slide_id>/augmentation/patch_views.h5",
    start=0,
    stop=16,
)
```

This lets the feature-extraction branch stream chunks rather than loading all
three views for an entire WSI into RAM at once.

## Resume, integrity and QC

The runner writes `manifest.csv` atomically after every slide. The run
fingerprint covers selected source identities, the inventory hash, pipeline
code hash, runtime versions and scientific settings. Resume additionally checks
the HDF5 SHA256, schema, metadata, coordinate alignment, quality tables and
requested QC files. Missing, stale or corrupt output is regenerated. A run lock
prevents concurrent writers, and each slide is installed from a staging
directory only after post-write validation. Use `--force` to recompute valid
results.

QC is saved by default. The montage samples patches across the tissue-percentage
distribution, while `augmentation_quality.csv` records intensity, saturation,
clipping, background and gradient metrics for every Original/Aug1/Aug2 image.
Use `--qc-sample-count` to change montage coverage or `--no-qc` for a non-review
run.

Independently verify the final artifacts and re-extract source patches with:

```bash
python scripts/verify_pilot14.py ../lung-wsi-mil-outputs/pilot14-final --verify-source-md5
```

For the final acceptance run, set `--source-sample-count` at least as high as
the largest per-slide patch count (or simply `100000`) to re-extract and compare
every stored Original patch, not only the default three evenly spaced
samples per slide:

```bash
python scripts/verify_pilot14.py ../lung-wsi-mil-outputs/pilot14-final \
  --verify-source-md5 --source-sample-count 100000
```

The augmentation master seed is converted into a stable per-slide seed, so a
slide's augmented views stay reproducible even if another slide fails or the run
is resumed later.
