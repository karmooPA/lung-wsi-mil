# Workspace and external data

Keep the source checkout, raw images and generated outputs in separate directories.
Their parent directory can be anywhere on a suitable local drive:

| Directory | Contents | Tracked |
| --- | --- | --- |
| `lung-wsi-mil/` | Source, tests, metadata, reference splits, docs and local `.venv/` | Source files only |
| `lung-wsi-mil-inputs/pilot14/` | External raw WSIs, with their GDC directories/companions intact | No |
| `lung-wsi-mil-outputs/pilot14-current/` | New-code pilot output and control files | No |
| `lung-wsi-mil-outputs/quality/` | New portable quality reports | No |
| `lung-wsi-mil-outputs/archive/` | Historical accepted outputs retained by the operator | No |

The delivered source archive does not contain raw WSIs, accepted image HDF5s,
weights, virtual environments or external output directories. Historical results
record their actual original execution locations; they are not current machine
configuration. Recreate an environment after extracting the checkout.

`metadata/pilot14_selection.csv` pins seven Normal and seven Tumor slides,
patient IDs, filenames, expected sizes and MD5 checksums. Its records match the
frozen inventory. Preserve these source files exactly.

## Run a new pilot

After installing the package into `.venv/`, from the repository root:

```bash
bash scripts/run_pilot14.sh

.venv/bin/python scripts/run_quality_checks.py \
  --pilot-output ../lung-wsi-mil-outputs/pilot14-current \
  --verify-source-md5 --source-sample-count 100000
```

The shell wrapper passes the frozen cohort CSV, seed 42, chunk size 8 and six QC
samples, with source MD5 checks. Optional environment overrides are:

| Variable | Default |
| --- | --- |
| `LUNG_WSI_PYTHON` | `<checkout>/.venv/bin/python` |
| `LUNG_WSI_INPUT_DIR` | `<parent>/<checkout-name>-inputs/pilot14` |
| `LUNG_WSI_OUTPUT_DIR` | `<parent>/<checkout-name>-outputs/pilot14-current` |

The Python CLI also works on Windows without the shell wrapper. With the
virtual environment active, run `python scripts/run_pilot14.py` and supply the
external input/output directories, `--inventory metadata/dataset_inventory.csv`,
`--selection-csv metadata/pilot14_selection.csv`, seed/chunk/QC settings and
`--verify-source-md5` as in the [pilot guide](../workflows/pilot14.md).

Inputs and outputs must not overlap; pilot/full-cohort output must stay outside
the source checkout. Preserve companion directories for multi-file WSI formats.
The quality runner's `--output-dir` controls its reports.

## Resume and historical evidence

Resume is valid only when source identity, scientific configuration, code/runtime
fingerprints and saved artifacts pass their checks. Code relocation/formatting
changes provenance. Start the reorganized pipeline in a new output location,
such as `pilot14-current`, rather than overwriting historical `pilot14-final`.
Use `--force` only when deliberately regenerating that output directory.

The [2026-10-02 pilot](../results/pilot14-2026-10-02.md) and
[2026-10-17 feature/AMIL acceptance](../results/amil-validation-2026-10-17.md)
describe earlier measured runs. Their old paths, code IDs and hashes remain
historical evidence. They do not assert that those data directories exist on
the machine where this archive is extracted.
