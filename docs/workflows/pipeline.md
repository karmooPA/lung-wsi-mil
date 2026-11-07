# Run the TCGA-LUSC pipeline

The pipeline command connects the existing stages for a full cohort run. It uses
the checked-in [694-slide inventory](../../metadata/dataset_inventory.csv),
[patient splits](../../splits/) and [AMIL reference configuration](../../configs/amil-reference.json).
Choose the starting point that matches the data you already have. Read the
[experiment protocol](experiments.md) before interpreting its test results.

## External data layout

Keep the checkout and its Python environment separate from `Data/`. Install the
project with `pip install -e ".[dev]"` after installing the appropriate PyTorch
and torchvision pair, as described in the [environment guide](amil-validation.md).
For modes that extract features, obtain `KimiaNetPyTorchWeights.pth` under the
[upstream terms](amil-validation.md#genuine-kimianet-weights).

```text
Data/
├── input/
│   ├── wsi/                      # Raw GDC slide folders and files
│   └── weights/
│       └── KimiaNetPyTorchWeights.pth
├── derived/
│   ├── patches/                  # Per-slide preprocessing and patch-view HDF5
│   ├── features/
│   │   └── kimianet.h5           # Provenance-checked KimiaNet cache
│   └── runs/<run-name>/          # New patches and features for a named run
└── output/
    └── <run-name>/               # Run status, training, test and artifacts
```

Download the matching GDC cohort and obtain the upstream weights separately
before running the `raw` mode. Preserve the GDC directory structure. Do not copy
the WSIs, weights, HDF5s, checkpoints or generated output into Git. The reference
inventory and splits remain in the checkout. Patient IDs are the grouping unit
even when one patient contributes slides of both labels.

## Choose the starting point

| Mode | Input required | Work performed |
| --- | --- | --- |
| `raw` | `input/wsi/` and official KimiaNet weights | Preprocess raw slides, save the three aligned patch views, extract KimiaNet features, then train and test AMIL. |
| `patches` | `derived/patches/` and official KimiaNet weights | Check the existing patch artifacts, extract features, then train and test. |
| `features` | `derived/features/kimianet.h5` | Validate the existing cache, then train and test. Preprocessing artifacts are needed for spatial overlays. |

Run one of the following commands from the installed environment. Use an unused
`--run-name`; it names the directory under `Data/output/`.

```powershell
# Windows: process raw WSI, then extract, train and test
lung-wsi-pipeline --data-root 'D:\Meraj Rastegar\Data' --run-name 'amil-from-raw' --start-at raw --generate-splits --device cuda

# Windows: reuse verified patch views, then extract, train and test
lung-wsi-pipeline --data-root 'D:\Meraj Rastegar\Data' --run-name 'amil-from-patches' --start-at patches --device cuda

# Windows: reuse the full feature cache, then train and test
lung-wsi-pipeline --data-root 'D:\Meraj Rastegar\Data' --run-name 'amil-from-features' --start-at features --device cuda
```

On Linux/WSL, use the same flags with a local path such as `/data/lung-wsi/Data`.
Use `--device cpu` in a CPU environment.

Omitting the start mode begins at raw WSI. The raw and patches modes write new
derived data under `derived/runs/<run-name>/` and the features mode reads the
accepted cache in `derived/features/`. The pipeline uses the checked-in patient
partitions by default. Add `--generate-splits` to create fresh grouped Train/CV
and test partitions and five inner folds per seed in `output/<run-name>/splits/`.
Use `--inventory`, `--splits` and `--config` to supply another compatible cohort and
protocol. Split generation uses the reference seeds.

Add `--check-only` to verify the selected inputs without creating a run. The raw
check matches slide coverage to the inventory and verifies the official weights;
raw MD5 checks run during processing. The patches check confirms artifact
presence and the official weights, and extraction validates patch content. The
features check verifies the full cache and the test slide geometry for requested
overlays. Repeat the original command with `--resume` to continue a named run
using the recorded inputs and configuration.

A cache from the 14-slide pilot cannot stand in for the 694-slide cache. The
preflight checks slide IDs, patient IDs, labels, feature shapes, finite values,
provenance and the saved splits. A rejected input does not silently shrink the
cohort or substitute synthetic features. A new run needs a separate output
directory; preserve completed runs and their recorded paths and checksums.

## Read the outputs

The wrapper writes `workspace.json` and `workspace-status.json` in the run folder.
The experiment outputs below are inside `output/<run-name>/experiment/`.

| Output | What it contains |
| --- | --- |
| `run.json`, `status.json` | Experiment identity, frozen configuration, input and code hashes, and progress. |
| `inputs/` | Frozen inventory and partition CSVs used by this run. |
| `training/seed_*/fold_*/` | Best and resumable epoch checkpoints, histories, and validation predictions and metrics. |
| `training/seed_*/selection.json` | The fold and epoch selected before evaluating its held-out patients. |
| `test/seed_*/` | Original-view predictions, metrics, patient-bootstrap confidence intervals, curves, and patch-level attention. The workspace command renders every held-out slide by default. |
| `test/summary.json`, `test/report.md` | The five primary test results and their descriptive aggregate. |

The pipeline trains 25 models: five outer patient splits, with five validation
folds inside each outer training partition. Training uses Original, Aug1 and
Aug2 as separate bags. It selects one checkpoint per seed on validation AUROC,
then validation loss, then fold number. Validation and the five primary test
evaluations use only Original views. The test threshold is fixed at 0.5; test
metrics do not select a fold, checkpoint or threshold. The outer test sets
overlap, so pooling their slides would not produce an independent test cohort.

For the frozen cohort the five seed holdouts contain 141, 139, 138, 141 and 140
slides. The workspace command defaults to `--heatmap-limit 0`, which renders all
699 seed/slide appearances. Use `--heatmap-limit 6` to render a fixed subset per
seed, or `--no-heatmaps` to keep the patch-level attention export alone.

The [November 3 result](../results/full-cohort-run-2026-11-03.md) records a real
CUDA run and its five primary tests. The [November 4 supplement](../results/all-fold-tests-2026-11-04.md)
tested all 25 frozen checkpoints on their respective seed holdouts without
retraining or changing the primary results. That historical supplement is not
part of the default pipeline. Source-aligned attention shows relative patch
weights within a slide; it does not identify tumor boundaries.

The dated reports retain their original execution paths. The server migration
preserves compatibility aliases for those paths. The November 3 primary run now
resides in `output/reference-20261103/experiment-reference/`; the November 4
supplement, reports and heatmap examples are in `output/all-fold-tests-20261104/`,
`output/reports-20261104/` and `output/heatmap-examples-20261104/`. Use a new run
name for new work. The migration did not change the recorded results or rename
the paths inside their frozen metadata.

The complete 141-slide seed-2 heatmap export lives in
`output/seed2-heatmaps-20261107/`, with its prediction index and per-slide images.
