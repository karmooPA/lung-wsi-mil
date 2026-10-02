# WSL workspace layout

The project has three sibling folders under `/home/mrage/Codes/Python`:

| Directory | Contents | GitHub |
| --- | --- | --- |
| `lung-wsi-mil` | Source code, tests, versioned metadata, documentation and local Python environment | Code/documentation/metadata are tracked; `.venv` is local |
| `lung-wsi-mil-inputs/pilot14` | The 14 raw WSI files copied from drive M in GDC subdirectories | Local only |
| `lung-wsi-mil-outputs` | Current pilot results, QC/test reports and archived runs | Local only |

The repository stays at its original path, so Git configuration and Python
environment paths remain valid. There are no raw WSI or runtime output folders
inside the repository. The original dataset remains on drive M.

`metadata/pilot14_selection.csv` pins the exact 7 Normal and 7 Tumor slides,
patient IDs, filenames, expected sizes and MD5 checksums. Those records match
the frozen dataset inventory. The pilot runner hashes the complete local source
files before processing and refuses mismatches.

## Reproduce or resume

From `lung-wsi-mil`:

```bash
bash scripts/run_pilot14.sh

.venv/lung-wsi-augmentation/bin/python scripts/run_quality_checks.py \
  --pilot-output ../lung-wsi-mil-outputs/pilot14-final \
  --verify-source-md5 --source-sample-count 100000
```

The shell wrapper uses the repository's Python environment, the pinned selection
CSV, the external input folder and the external final-output folder. It forwards
extra pilot arguments, so `bash scripts/run_pilot14.sh --force` regenerates the
pilot. Optional overrides are `LUNG_WSI_PYTHON`, `LUNG_WSI_INPUT_DIR` and
`LUNG_WSI_OUTPUT_DIR`. Defaults are resolved from the repository location.

All routine CLI reports default to the sibling output folder. The quality
runner's `--output-dir` controls its validation, audit and summary outputs.

## Preserved historical runs

`lung-wsi-mil-outputs/archive` contains the baseline run, intermediate hardened
run, previous accepted run, its quality reports, baseline ZIP and PDF inspection
artifacts. These are historical snapshots; their recorded absolute paths and
generating-code fingerprints are preserved as provenance. Use the current
`pilot14-final` directory for resume, downstream consumption and verification.

The current run is regenerated in the new directories after relocation. Its
configuration, source identities, HDF5 metadata and manifest all refer to the
external paths. Comparing every image, coordinate and tissue array against the
previous accepted snapshot confirms scientific reproducibility.
