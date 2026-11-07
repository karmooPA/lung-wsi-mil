# Roadmap implementation — 2026-11-01

The **5-Fold training foundation is authored and committed by karmooPA**.
The remaining roadmap integration, evaluation, visualization and independent
verification are attributed to **mragetsars**. See the corrected
[ownership record](ownership.md) and [commit map](roadmap-attribution-correction.json).

| Roadmap item | Implemented behavior |
| --- | --- |
| 5-Fold Training | Five frozen patient-aware folds, epoch validation, early stopping and selected/resumable checkpoints |
| 5 Seeds | Reference seeds 2, 3, 5, 7, 11; 25 fits, progress, integrity checks and input/code/runtime fingerprints |
| Final Test | One validation-selected fold/checkpoint per seed, Original only, no optimizer or test-based selection |
| AUC / Metrics | Exact predictions, AUROC/AP, classification/calibration metrics, patient bootstrap and PNG/SVG figures |
| Attention / Heatmap | Raw per-patch attention, source-aligned Level-0 raster maps, coverage and thumbnail overlays |

The end-to-end software test exercises all 25 fits, held-out evaluation, raw
attention and spatial overlays on synthetic data. A separate interruption test
checks matching epoch histories and model weights after resume. Other tests
reject leakage/identity corruption, undefined metrics, stale protocol/checkpoints
and coordinate mismatches. The dated [software validation report](../results/roadmap-software-validation-2026-11-01.md)
records 276 passed tests and the completed public synthetic CLI run; detailed
logs and labelled sample figures accompany the handoff.

At the time of the November 1 software check, real-data execution remained a
separate step. The [November 3 real run](../results/full-cohort-run-2026-11-03.md)
later completed all 25 fits and the five primary held-out tests; the
[November 4 supplement](../results/all-fold-tests-2026-11-04.md) evaluated all
25 frozen checkpoints separately. The repository contains the frozen metadata
and reports; the real feature cache, raw WSIs and pretrained weights remain
external. The November 1 smoke test itself reports synthetic results only.

See [the pipeline guide](../workflows/pipeline.md), the
[experiment protocol](../workflows/experiments.md) and the explicit
[reference configuration](../../configs/amil-reference.json).
