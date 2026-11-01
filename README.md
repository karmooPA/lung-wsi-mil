# Lung WSI MIL

Reproducible binary **Tumor / Normal** classification research for TCGA-LUSC
whole-slide images, using frozen KimiaNet features and attention-based MIL.

The frozen cohort contains **694 slides, 347 per class, from 373 patients**.
Some patients have both Normal and Tumor tissue. All slides and derived views
from one patient must remain in the same data partition.

## Pipeline and current scope

1. Audit the frozen GDC manifest, labels, identity, and available image evidence.
2. Detect tissue and extract 512 × 512 patches at effective 5× magnification.
3. Keep patches with strictly more than 20% tissue; no arbitrary slide patch-count cutoff.
4. Store aligned Original / Aug1 / Aug2 images in lossless, chunked HDF5.
5. Stream all three views through genuine, frozen KimiaNet into 1024-dimensional embeddings.
6. Load patient-grouped partitions, remove padding, and optimize AMIL view-bags.

| Component | Available implementation and evidence |
| --- | --- |
| Dataset audit | Offline replay and explicit local scans with source provenance |
| Preprocessing / augmentation | Bounded pilot/full-cohort runners, resume checks, independent verification |
| KimiaNet cache | Genuine checkpoint validation and provenance-bound streaming extraction |
| MIL data | HDF5 Dataset, padded collate, grouped splits, five-seed / five-fold reference CSVs |
| AMIL | Three-output model, initialization, numerical and integration tests |
| Training | Complete five-fold × five-seed AMIL runner, validation selection, early stopping and resumable checkpoints |
| Final test / metrics | Original-only held-out evaluation, AUROC/AP, patient bootstrap and reproducible PNG/SVG reports |
| Attention / heatmaps | Every test slide's raw attention plus source-aligned maps, coverage and thumbnail overlays |
| Historical real data | [14-WSI / 42-view acceptance](docs/results/amil-validation-2026-10-17.md) |
| Execution scope | Full synthetic software acceptance is available; new real-cohort training requires external features and preprocessing artifacts |

The historical acceptance checks an initialized model's interface and numerical
behavior. It does not establish trained classification performance. Raw WSIs,
KimiaNet weights and real feature caches are external inputs.

## Install

Use **Python 3.12** on 64-bit Linux/WSL or Windows. From this checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

On Windows use `py -3.12 -m venv .venv` and activate
`.venv\Scripts\Activate.ps1`. The
[AMIL environment guide](docs/workflows/amil-validation.md) has CUDA commands.
Install the chosen torch/torchvision wheel pair before installing the project extra.

| Install target | Contents |
| --- | --- |
| `pip install -e .` | Preprocessing dependencies |
| `pip install -e ".[audit]"` | Frozen-source/local WSI audit support |
| `pip install -e ".[augmentation]"` | Three-view transforms and image HDF5 storage |
| `pip install -e ".[mil]"` | Augmentation, feature extraction, grouped splits and MIL |
| `pip install -e ".[dev]"` | Complete runtime, pytest, Ruff and packaging tools |

Pins are defined in `requirements/` and consumed by `pyproject.toml`.
These files are dependency components; the extras combine them without duplicated pins.
Offline audit functions use the standard library. A dependency-free source invocation
is `PYTHONPATH=src python -m lung_wsi_mil.audit.inventory --help`.

## Verify

```bash
python scripts/run_quality_checks.py
python -m build
```

The quality runner checks formatting, lint, compilation, tests, frozen sources,
recorded audit replay, dependency consistency, commit policy and whitespace.
Reports go to `../lung-wsi-mil-outputs/quality/` by default. A plain
`python -m pytest` discovers only `tests/`.

Real-data acceptance is separate and fails when its input is missing:

```bash
python scripts/validate_amil.py --pilot14-h5 /external/pilot14-kimianet.h5 --output ../lung-wsi-mil-outputs/amil-validation.json
```

Source relocation changes code/provenance fingerprints. Keep accepted historical
outputs as evidence and use a new output location for the reorganized pipeline;
old caches must not be made current by editing their fingerprints.

## Run

```bash
lung-wsi-validate-sources
lung-wsi-audit --mode recorded
lung-wsi-splits --output-dir ../lung-wsi-mil-outputs/new-splits

lung-wsi-pilot /external/pilot14 ../lung-wsi-mil-outputs/pilot14-new --inventory metadata/dataset_inventory.csv --selection-csv metadata/pilot14_selection.csv --verify-source-md5
lung-wsi-verify-pilot ../lung-wsi-mil-outputs/pilot14-new --verify-source-md5

lung-wsi-process /external/TCGA-LUSC ../lung-wsi-mil-outputs/cohort-new --inventory metadata/dataset_inventory.csv --workers 6 --verify-source-md5
lung-wsi-extract --help
```

Commands accept `--help`. Existing `scripts/*.py` launchers remain available;
their importable implementations now live in `lung_wsi_mil`. Old Python imports
under `src.*` should be changed to `lung_wsi_mil.*`.

## Complete the training/evaluation roadmap

```bash
lung-wsi-experiment validate --features /external/features/full-kimianet.h5 --config configs/amil-reference.json
lung-wsi-experiment run --features /external/features/full-kimianet.h5 --config configs/amil-reference.json --device cuda --patch-root /external/TCGA_LUSC_HDF --output ../lung-wsi-mil-outputs/amil-reference
```

Use `--device cpu` with the CPU runtime. `train` and `test` can run separately;
`--resume` verifies/reuses the same experiment and continues interrupted epochs.
The default trains all 25 models, selects on validation only and evaluates one
frozen model per seed. Validation/test use Original views only. All outputs stay
outside Git; the source model, preprocessing rules and frozen splits are preserved.

For a complete software check without real data:

```bash
lung-wsi-experiment smoke --output ../lung-wsi-mil-outputs/software-smoke
```

That creates synthetic inputs and runs every stage; its metrics are not real
TCGA results. See the [experiment protocol and server commands](docs/workflows/experiments.md),
[roadmap implementation status](docs/development/roadmap-completion.md), and
[reference config](configs/amil-reference.json).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/lung_wsi_mil/` | Installable audit, augmentation, data, features, MIL, preprocessing and workflows |
| `scripts/` | Compatibility launchers, quality/acceptance runners and Git tools |
| `tests/` | Portable tests and explicit optional real-data/device checks |
| `metadata/source/` | Byte-preserved frozen source documents |
| `metadata/` | Committed cohort inventory and recorded evidence summaries |
| `splits/` | Frozen patient-aware reference partitions |
| `configs/` | Explicit, frozen experiment defaults |
| `requirements/` | Single-source pinned dependency components |
| `docs/` | [Guide index](docs/README.md), architecture, workflows and dated results |

## Contributors and Git handoff

- **karmooPA**: original preprocessing, feature APIs, MIL data/splits, AMIL and five-fold training with resumable checkpoints.
- **mragetsars**: cohort audit, augmentation, five-seed orchestration, final evaluation/attention, feature-cache provenance and acceptance.

See [ownership evidence](docs/development/ownership.md),
[contribution conventions](CONTRIBUTING.md), and the
[history rewrite / force-push guide](docs/development/git-history.md).
The delivered checkout includes its complete `.git` directory. No remote push
was performed during organization.

The source code retains its original [MIT license](LICENSE). KimiaNet weights
retain their separate upstream agreement, linked in the AMIL guide.
