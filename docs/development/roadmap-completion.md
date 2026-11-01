# Roadmap implementation — 2026-11-01

This extension is authored by **mragetsars**, as requested. Existing karmooPA
and mragetsars work and its history remain intact.

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
and coordinate mismatches. Final execution evidence is in the handoff reports.

**Real-data execution remains a separate run.** The supplied repository contains
frozen metadata and historical acceptance evidence, not the full real feature
cache, raw WSIs or pretrained weights. The new smoke test does not replace that
evidence and provides no real diagnostic AUC. The implementation is ready for
the documented server commands once those external inputs are supplied.

See [the complete experiment guide](../workflows/experiments.md) and the explicit
[reference configuration](../../configs/amil-reference.json).
