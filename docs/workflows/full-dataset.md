# Full-dataset preprocessing and augmentation

`scripts/run_wsi_dataset.py` runs the existing preprocessing and augmentation
operators on **every supported WSI physically present** under the input root.
It resolves files against the inventory and rejects unknown files, ambiguous
coverage, duplicate slide IDs or physical paths, and invalid labels. It does
not impose the pilot's 14-slide limit or class balance. Multiple slides from the
same patient are allowed; later training/validation/test splits must still be
made at patient level.

The scientific defaults remain 5x magnification, 512x512 patches, tissue
threshold 20%, color-distance threshold 35, thumbnail size 1024, seed 42,
augmentation chunk size 8 and six QC montage samples per slide. Every valid
patch receives Original, Aug1 and Aug2. A slide with zero valid patches remains
represented in the manifest with an empty HDF5 and
`feature_extraction_ready=False`.

## Run

Keep inputs and outputs outside the repository in disjoint directories. For a new run, the following PowerShell example uses external input/output
paths and a prepared environment:

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_wsi_dataset.py `
  'D:\Data\TCGA_LUSC_WSI' `
  'D:\Outputs\TCGA_LUSC_HDF_Current' `
  --inventory metadata/dataset_inventory.csv `
  --workers 6 --verify-source-md5 `
  --seed 42 --chunk-size 8 --qc-sample-count 6
```

The equivalent command with generic paths and a prepared Python environment is:

```bash
python scripts/run_wsi_dataset.py /data/TCGA-LUSC /outputs/TCGA-LUSC-HDF \
  --inventory metadata/dataset_inventory.csv \
  --workers 6 --verify-source-md5 \
  --seed 42 --chunk-size 8 --qc-sample-count 6
```

`--workers` defaults to 1. Six workers process six independent slides at most,
with one Torch thread per worker. The parent alone updates cohort control
files. Images are transformed in bounded chunks and written incrementally;
coordinates and per-view numeric QC rows remain in memory per slide. The
runner checks available space before each slide and before image writing,
including the uncompressed image estimate and a 2 GiB reserve
(`--min-free-gib`). On Windows it temporarily requests that the system remain
awake during the run and restores the request on exit.

`--verify-source-md5` validates each source against its inventory MD5 before
processing or resuming that slide. A slide failure is recorded while the other
slides continue. The runner exits with code 0 only when every selected slide
succeeds; recorded slide failures produce code 2.

## Outputs and progress

```text
TCGA_LUSC_HDF/
├── run_config.json
├── selection.csv
├── manifest.csv
├── status.json
├── failures/<slide_id>.json       # Present only for recorded failures
└── slides/<slide_id>/
    ├── complete.json
    ├── preprocessing/
    │   ├── coordinates.npy
    │   ├── valid_patches.csv
    │   ├── result.json
    │   └── qc/...
    └── augmentation/
        ├── patch_views.h5
        ├── augmentation_quality.csv
        ├── augmentation_quality_summary.json
        ├── augmentation_qc_patch_0000.png
        └── augmentation_qc_montage.png
```

`manifest.csv` includes every selected slide from the start and records its
status, patch count, final HDF5 path, size, SHA256 and any error.
`status.json` reports completed/total, successes, failures, resumed slides,
active slide IDs and remaining disk space. `run_config.json` records scientific
settings, code/runtime versions, source identities and per-slide fingerprints.

Each `patch_views.h5` preserves the pilot format:

- `images`: uint8 `[3, N, 512, 512, 3]`, ordered Original / Aug1 / Aug2;
- `coords`: int64 `[N, 2]`, matching Level-0 coordinates;
- `tissue_percentage`: float32 `[N]`.

Gzip-compressed RGB storage reconstructs the pipeline's normalized float32
tensors exactly. Use `lung_wsi_mil.augmentation.storage.load_augmented_chunk` for
bounded downstream reads. See [pilot-14-wsi.md](pilot14.md) for the shared
HDF5 layout and loader example.

## Resume and independent verification

Run the same command again to resume. Each slide has its own fingerprint, so
adding other slides or changing the worker count does not invalidate completed
slides. Changes to that slide's source identity, scientific settings, QC
settings, generating code or runtime do invalidate it. Resume checks hashes,
schema, preprocessing/augmentation alignment, every numeric QC row and the
requested QC images. Each newly generated slide is validated before its staged
directory is installed. `--force` regenerates otherwise valid results.

After the runner finishes, independently verify every artifact, rehash every
source, and compare stored Original images to source re-extractions:

```powershell
& '.\.venv\Scripts\python.exe' scripts/verify_wsi_dataset.py `
  'D:\Outputs\TCGA_LUSC_HDF_Current' `
  --verify-source-md5 --source-sample-count 100000 --workers 6 --threads 1
```

```bash
python scripts/verify_wsi_dataset.py /outputs/TCGA-LUSC-HDF \
  --verify-source-md5 --source-sample-count 100000 --workers 6 --threads 1
```

The default verifier compares three Original samples per slide. To compare
**every** Original patch, set `--source-sample-count` at least as high as the
largest per-slide patch count; the example uses 100000. Check that each reported
`source_reextract_samples` equals its `patch_count`. Verification writes
`verification.json` and `verification.md`; only a `PASSED` report and exit code
0 confirm completed independent validation.

An abrupt termination can leave `.dataset.lock` and hidden `.working.*` or
`.backup.*` directories. Before removing a stale lock or recovering staging
files, confirm the recorded parent and worker processes are inactive. Do not
delete these paths automatically: a backup may hold a previous valid result.
An interrupted slide restarts from scratch; other completed slides can resume.
The verifier rejects leftover temporary/staging/backup artifacts and failure
records, so recovery must account for them before final acceptance.
