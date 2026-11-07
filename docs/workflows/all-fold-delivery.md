# Deliver all 25 seed and fold evaluations

Use a completed, sealed training run with the reference seeds 2, 3, 5, 7 and 11
and five patient folds per seed. Keep training, staging and preprocessing outside
the checkout. The [workspace workflow](pipeline.md) describes input checks,
patient partition generation, preprocessing and official KimiaNet extraction.

To train without first running the five primary tests, use the experiment
command with the validated feature cache and patient partitions.

    lung-wsi-experiment train --features <feature-cache> --inventory <inventory.csv> --splits <splits-dir> --config configs/amil-reference.json --device cuda --output <training-run>

Once training is sealed, evaluate every best fold checkpoint on the outer test
set of its seed and export every slide overlay.

    lung-wsi-deliver-folds --run-dir <training-run> --patch-root <patch-root> --output <delivery-staging>

The output staging path holds an identity, status, summary and 25 directories
named seed_02_fold_01 through seed_11_fold_05. Use a fresh staging directory.
To continue an interrupted export, repeat the command with --resume and the
same rendering size. Completed directories are verified before reuse. Keep
staging separate from the final output so that its run metadata and unfinished
folders do not appear among the 25 published results.

Each pair directory contains:

| Path | Contents |
| --- | --- |
| train/ | Frozen checkpoints, epoch history, validation metrics and predictions |
| test/predictions.csv | One Original-view prediction per test slide |
| test/metrics.json and test/patient_bootstrap.json | Slide metrics and patient-cluster AUROC interval |
| test/curves.json and test/metrics.png | ROC, precision–recall and confusion matrix |
| test/attention/ | Per-patch weights and level-zero coordinates |
| test/heatmaps/ | Index and one overlay, color map, coverage map and metadata record per test slide |
| report.md and learning-curves.png | Analysis, linked evidence and learning curves |
| complete.json | Checksums and pair identity |

The five models within each seed share its outer holdout. The full reference
cohort produces 3,495 model/slide predictions and overlays. The validation-selected
model for each seed is flagged for interpretation. Test outcomes do not select
folds, epochs or thresholds.

After validation, publish the 25 pair directories as the only entries in the
final output directory. Keep the staging manifest, run summary and validation
reports beside the archived training execution.
