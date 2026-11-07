# AMIL experiment: five folds × five seeds

This workflow implements the remaining roadmap: **5-Fold Training → 5 Seeds →
Final Test → AUC / Metrics → Attention / Heatmap**. It uses the existing frozen
patient-aware partitions and the unchanged AMIL three-output architecture.

The [November 3 CUDA run](../results/full-cohort-run-2026-11-03.md) trained 25
models on the full 694-slide cohort and evaluated the five validation-selected
checkpoints. The [November 4 supplementary run](../results/all-fold-tests-2026-11-04.md)
also evaluated all 25 frozen checkpoints without changing the primary result.
For a new real run, use the [pipeline workflow](pipeline.md) or supply the
external full feature cache and preprocessing artifacts to the commands below.

## Freeze the protocol before looking at test results

| Decision | Fixed policy |
| --- | --- |
| Outer repetitions | Seeds `2, 3, 5, 7, 11`, using committed `splits/seed_*/` CSVs |
| Inner validation | Five patient-disjoint folds within each seed's Train/CV set |
| Fits | 25 independently initialized AMIL models for the complete reference run |
| Training input | Original, Aug1 and Aug2 as separate bags of the same slide |
| Validation / test | Exactly one Original bag per slide; no augmented test-time views |
| Checkpoint selection | Highest validation AUROC; exact AUROC ties use lowest validation loss; earlier identical results are retained |
| Early stopping | `patience` epochs without a meaningful AUROC improvement, or loss improvement at exactly tied AUROC |
| `min_delta` | Controls meaningful progress/patience resets; never trades a lower AUROC for better loss |
| Final model per seed | Best selected validation AUROC, then validation loss, then lowest fold index |
| Final test | Only the selected checkpoint on that seed's held-out test patients |
| Classification threshold | Fixed in the saved config, default `0.5`, equality is positive |
| Test influence | No test predictions or metrics are used for epochs, folds, thresholds or optimizer settings |

This follows the project roadmap's best-validation-fold protocol. It is not a
five-model test ensemble or a refit on all Train/CV data. Those alternatives
would be different, separately declared experiments. Selected-epoch OOF
validation predictions are saved for auditing; they are model-selection data,
not the final performance estimate.

## Inputs and preflight

The cache needs every inventory slide, with float32 `[3,N,1024]` embeddings,
nonnegative int64 `[N,2]` Level-0 coordinates, a binary scalar label and its
patient/case ID. Empty bags, nonfinite values, missing/extra slides, duplicate
IDs and metadata conflicts fail before fitting. Every partition must contain
both classes and preserve patient separation. Each Train/CV slide must appear
in exactly one inner validation fold.

By default the cache must have complete feature provenance: the official
KimiaNet source/checkpoint identity, its frozen inventory hash, self-consistent
config/code/source fingerprints and per-group embedding/coordinate hashes.
The training preflight verifies cache integrity and recorded provenance, not
the unavailable raw WSI bytes or a fresh encoder re-extraction. Use the separate
image/feature acceptance procedures when those checks are needed.

The old 14-slide pilot feature file is insufficient for the full 694-slide
experiment. To extract the full cohort, omit the pilot selection argument:

```bash
lung-wsi-extract --patch-root /external/TCGA_LUSC_HDF --inventory metadata/dataset_inventory.csv --weights /external/weights/KimiaNetPyTorchWeights.pth --output /external/features/full-kimianet.h5 --batch-size 8 --device cuda
```

Then inspect all experiment inputs without fitting:

```bash
lung-wsi-experiment validate --features /external/features/full-kimianet.h5 --inventory metadata/dataset_inventory.csv --splits splits --config configs/amil-reference.json
```

`--allow-unverified-features` is an explicit escape hatch for synthetic or legacy
cache software tests. Its outputs are marked `SYNTHETIC_OR_UNVERIFIED` and cannot
be reported as genuine full-cohort acceptance. The synthetic generator never
adds fake official provenance.

## Run all stages

Install Python 3.12 and the selected compatible torch/torchvision wheel pair,
then `python -m pip install -e ".[dev]"`. CUDA commands and weight terms are in
[the feature/AMIL guide](amil-validation.md).

```bash
lung-wsi-experiment run \
  --features /external/features/full-kimianet.h5 \
  --inventory metadata/dataset_inventory.csv --splits splits \
  --config configs/amil-reference.json --device cuda \
  --patch-root /external/TCGA_LUSC_HDF \
  --output ../lung-wsi-mil-outputs/amil-reference
```

Use `--device cpu` for the CPU environment. `scripts/run_experiment.py` is the
equivalent repository launcher. Windows PowerShell example:

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_experiment.py run `
  --features 'D:\Features\full-kimianet.h5' `
  --inventory metadata/dataset_inventory.csv --splits splits `
  --config configs/amil-reference.json --device cuda `
  --patch-root 'D:\Data\TCGA_LUSC_HDF' `
  --output 'D:\Outputs\amil-reference'
```

Outputs must be outside the checkout and must not contain the input feature
file. Existing output is refused unless `--resume` is supplied for exactly the
same experiment. The config, inventory, all requested split files, feature
content hash, code hashes and runtime/device settings are frozen in `run.json`.
The input CSV bytes are copied into `inputs/` so the test stage uses the exact
frozen partitions, rather than a later edited repository file.

## Separate training and final testing

Use `train` to finish and freeze all fold selections before test inference:

```bash
lung-wsi-experiment train --features /external/features/full-kimianet.h5 --config configs/amil-reference.json --device cuda --output ../lung-wsi-mil-outputs/amil-reference
lung-wsi-experiment test --run-dir ../lung-wsi-mil-outputs/amil-reference --patch-root /external/TCGA_LUSC_HDF
```

The test subcommand does not accept training hyperparameters or a new threshold.
An optional `--features` path can point to a moved copy with the exact training
hash. A completed test is not evaluated again; `test --resume` verifies and
returns its frozen artifacts. Partial completed seeds are checked and reused.

Without `--patch-root`, scalar metrics, figures and raw attention still export
for every test slide. Spatial heatmaps are marked `not_requested`; they need
actual source dimensions and patch footprints. A later separate heatmap render
can use the saved attention tables without retraining or changing test results.

## Resume and integrity

```bash
lung-wsi-experiment run --features /external/features/full-kimianet.h5 --config configs/amil-reference.json --device cuda --patch-root /external/TCGA_LUSC_HDF --output ../lung-wsi-mil-outputs/amil-reference --resume
```

Each epoch atomically saves `last.pt`: current model/optimizer state, selected
best model, scores/history, patience state, Python/NumPy/Torch RNG state and
separate loader generators. `best.pt` and history are regenerated from that
source of truth after an interruption. Loading uses `weights_only=True`.
The interrupted-versus-uninterrupted CPU test verifies identical selected
weights and epoch history.

Completed fold and test stages have checksummed file manifests. Missing,
modified or extra completed artifacts fail closed. Configuration, input, code,
device or runtime changes require a new run; changing `epochs` is not an
interruption resume. A run lock prevents concurrent writers. Normal interrupts
release it; after a hard process kill, establish that no writer remains before
removing its stale `run.lock`. Temporary stage files are not complete results.

The software acceptance exercised CPU, and the recorded real full-cohort run
used CUDA. MPS is available when the runtime supports the requested deterministic
operations. Deterministic settings do not promise identical floating-point
results across different hardware or library versions.

## Metrics and figures

For each seed, save one prediction per held-out slide, with patient ID, true
label, Tumor probability, threshold/prediction, seed/fold/checkpoint identity,
and config/cohort/split/feature hashes. The report includes:

- AUROC and non-interpolated average precision (AP); AP is not trapezoidal PR AUC.
- Accuracy, balanced accuracy, sensitivity, specificity, precision, NPV and F1.
- Brier score, log loss, class prevalence and a labelled confusion matrix.
- Patient-cluster percentile bootstrap 95% AUROC interval; every sampled
  patient's slides move together and one-class resamples are counted/skipped.
- ROC/PR curves, confusion figures and repeated-split ROC in PNG/SVG.
- Mean and sample SD over seeds, with defined-value counts. Repeated seed
  holdouts overlap; combined predictions are not pooled as independent samples.

Undefined rates/one-class AUROC are null with an explicit reason, never a fake
zero. Preflight rejects one-class validation/test partitions for this protocol.
The combined prediction CSV is for traceability, not independent-cohort scoring.

## Attention and spatial heatmaps

Every final-test Original bag exports its exact patch coordinates and attention
weights to a CSV plus provenance JSON. Weight order follows the feature cache;
padding is excluded, all weights are finite/nonnegative and sum to one.

With `--patch-root`, maps read each source's `preprocessing/result.json`,
`coordinates.npy` and QC thumbnail. The full-slide Level-0 dimensions and
physical patch footprint drive rasterization. Coordinates/order must match
the feature source. Sampled patch rectangles are mapped on both axes, clipped
at slide edges and averaged where they overlap. There is no smoothing or
extrapolation into unsampled regions: those pixels are NaN in `attention_raw.npy`
and transparent in `heatmap.png`. Coverage counts/masks are also saved.

Display uses within-slide average-rank attention percentiles, with constant
attention at the neutral midpoint. Absolute raw weights remain in the CSV and
numeric map. This is model attention, not a tumor segmentation mask or a
histological ground truth. Scales should not be compared as absolute evidence
across different bags.

By default, six deterministic, label-balanced examples per seed are rendered;
all raw attention is exported. Set `heatmap_limit` to `0` for every test slide.
Without a saved thumbnail, a geometry-only heatmap/coverage panel is produced
and `overlay_generated` is false. The thumbnail is never fabricated.

Standalone rendering:

```bash
lung-wsi-heatmap --attention /external/run/test/seed_2/attention/SLIDE.csv --preprocessing /external/patches/slides/SLIDE/preprocessing --output /external/maps/SLIDE
```

## Reproduce the software check without real data

```bash
lung-wsi-experiment smoke --output ../lung-wsi-mil-outputs/software-smoke
```

This creates 30 synthetic mixed-label patients / 60 slides, runs all 25 fits for
two epochs, performs the five final tests and writes synthetic overlays. It is
clearly labelled software verification. Its AUC must not be presented as a TCGA
result. The output folder must be new; it does not alter source metadata/splits.

## Key output paths

| Path under the run directory | Meaning |
| --- | --- |
| `run.json`, `status.json` | Frozen provenance/protocol and current progress |
| `inputs/` | Exact inventory and split snapshots |
| `training/seed_*/fold_*/last.pt` | Resumable epoch state |
| `training/seed_*/fold_*/best.pt` | Validation-selected weights |
| `training/seed_*/fold_*/history.json` | Epoch train/validation history |
| `training/seed_*/selection.json` | Final fold selected before testing |
| `training/seed_*/validation_oof_predictions.csv` | Selected-epoch validation audit |
| `test/seed_*/predictions.csv`, `metrics.json` | Held-out Original-only outputs |
| `test/seed_*/patient_bootstrap.json`, `curves.json` | Uncertainty and exact curves |
| `test/seed_*/attention/` | Every held-out slide's coordinate/weight CSV |
| `test/seed_*/heatmaps/` | Requested spatial maps, overlays and coverage |
| `test/summary.json`, `report.md` | Per-seed results and descriptive aggregate |

## Primary implementation references

- [PyTorch 2.14 reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- [scikit-learn 1.8 AUROC](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_auc_score.html)
- [OpenSlide Python: Level-0 coordinates and thumbnails](https://openslide.org/api/python/)

The saved project roadmap's training/evaluation sections informed the Original-only
evaluation, validation-only selection and patient-cluster bootstrap policies.
The configurable optimizer/epoch defaults are engineering choices, not asserted
values recovered from the paper.
