# Architecture and contracts

## Research target

The model predicts Tumor (`1`) versus Solid Tissue Normal (`0`) for a WSI from
the frozen TCGA-LUSC cohort. Normal tissue belongs to this cancer cohort; it is
not a healthy-donor category. The source summary contains 694 slides, 373 patients
and 153 patients with both tissue labels. Grouping by patient is mandatory.

## Boundaries

| Module | Responsibility | Persistent interface |
| --- | --- | --- |
| `audit` | Frozen metadata replay and new read-only image scans | CSV inventories, JSON evidence |
| `preprocessing` | Physical-scale resolution, tissue detection and patch QC | Level-0 coordinates, result JSON and optional QC |
| `augmentation` | Original / Aug1 / Aug2 with reproducible RNG | Lossless chunked RGB HDF5 |
| `features` | Genuine frozen KimiaNet, bounded streaming and provenance | One feature HDF5 group per slide |
| `data` | Dataset, padding, patient partitions and reference folds | Five-return Dataset / six-return collate |
| `mil` | AMIL, initialization, padding removal, loss and training | Three-return model and scalar training metrics |
| `experiments` | Frozen 5×5 fitting, checkpoint selection/resume and final testing | Config/input snapshots, selected weights and stage manifests |
| `evaluation` | Slide-level metrics and patient-cluster uncertainty | Exact prediction/metric/curve JSON and CSV |
| `visualization` | Headless metric figures and source-aligned attention | PNG/SVG plots, raw weights, coverage and spatial overlays |
| `workflows` | Cohort selection, locking, staging, resume and independent verification | Per-run control files and per-slide artifacts |

The package owns reusable implementation. Scripts only expose legacy commands
or repository-dependent quality/acceptance tools. Outputs and heavy inputs live
outside the checkout. An installed wheel contains algorithms; pass metadata
paths explicitly when no source checkout is available.

## Frozen scientific interfaces

| Stage | Shape / rule |
| --- | --- |
| Patch | RGB `512 × 512`, effective 5×, strictly >20% tissue |
| Augmentation tensor | float32 `[3, N, 3, 512, 512]`, ImageNet normalized |
| Stored image tensor | uint8 `[3, N, 512, 512, 3]`, losslessly restorable |
| Feature tensor | `[3, N, 1024]`, coordinates `[N, 2]` in Level-0 space |
| Dataset | `(embeddings, coordinates, label, slide_id, case_id)` |
| Collate | `(padded_bags, coordinates, labels, slide_ids, case_ids, is_augmented)` |
| Padding | Complete `-Inf` embedding rows; coordinates `[-1, -1]` |
| AMIL input | One finite, unpadded `[N, 1024]` bag, `N >= 1` |
| AMIL output | `(Y_prob, Y_hat, A)` with `[1, 1]`, `[1, 1]`, `[1, N]` |

Each of the three views remains a distinct training bag; the original collate
API is preserved. The experiment runner explicitly selects Original-only
validation/test views, fixes its threshold/config before testing, and selects
checkpoints/folds solely on validation. See the [experiment protocol](workflows/experiments.md).

## Reproducibility

Reference seeds are `2, 3, 5, 7, 11`. For each seed, a grouped stratified outer
split holds out approximately 20% of slides while keeping patients intact.
Five grouped folds are then generated within Train/CV with random state `42`.
Saved reference CSVs are immutable inputs, not runtime scratch output.

Resume checks bind code, source identity and settings. Reorganization changes
source paths and bytes, hence fingerprints. New runs need new output directories;
historical outputs remain associated with their generating code. The supplied
historical acceptance proves 14 slides / 42 view forwards; no fresh real-data
acceptance, trained checkpoint or diagnostic metric is asserted by this cleanup.

The 2026-11-01 roadmap extension adds the full experiment/heatmap implementation. Its synthetic software checks do not constitute a new real-cohort acceptance or a reported diagnostic metric.
