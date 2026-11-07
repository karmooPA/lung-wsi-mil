# Final full-cohort AMIL results — 2026-11-03

**COMPLETED: 25 training fits and five held-out test evaluations.** Mean test
AUROC across seeds is **0.974028 ± 0.021111** (descriptive mean ± sample SD).
The controller finished successfully at `2026-11-03T11:39:27.880284Z`
(15:09:27 Asia/Tehran). The result was inspected and its retained artifact hashes
verified on 2026-11-04.

## Held-out results

Metrics use **one Original view per slide**, with Tumor positive and a fixed
classification threshold of 0.5. Patients are disjoint between training,
validation and test within each seed. The selected fold/epoch was determined
from validation results before test inference.

| Seed | Test slides / patients | Selected fold / epoch | AUROC [95% CI] | Accuracy | F1 | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 141 / 74 | 3 / 19 | 0.995976 [0.9872, 1.0000] | 0.978723 | 0.979021 | 0.985915 | 0.971429 |
| 3 | 139 / 79 | 2 / 4 | 0.985300 [0.9689, 0.9980] | 0.949640 | 0.948905 | 0.928571 | 0.971014 |
| 5 | 138 / 71 | 1 / 12 | 0.975425 [0.9487, 0.9953] | 0.949275 | 0.948905 | 0.942029 | 0.956522 |
| 7 | 141 / 73 | 5 / 24 | 0.973642 [0.9418, 0.9961] | 0.943262 | 0.939394 | 0.885714 | 1.000000 |
| 11 | 140 / 75 | 1 / 39 | 0.939796 [0.8960, 0.9763] | 0.914286 | 0.913043 | 0.900000 | 0.928571 |

Each AUROC interval is a **patient-cluster percentile bootstrap** interval for
that seed's slide-level test result, with 1,000 requested/valid resamples and
95% confidence. These are not patient-level classification metrics or an
interval for the across-seed mean.

| Metric | Mean across five seeds | Sample SD |
| --- | --- | --- |
| AUROC | 0.974028 | 0.021111 |
| Average precision | 0.976951 | 0.021787 |
| Accuracy | 0.947037 | 0.022928 |
| F1 | 0.945854 | 0.023648 |
| Sensitivity | 0.928446 | 0.039123 |
| Specificity | 0.965507 | 0.025981 |

These are descriptive repeated-split results, not five independent cohorts.
The combined prediction export has **699 seed/slide rows covering 466 unique
slides**; the same slide can appear in several seeds. No pooled independent-test
AUROC is claimed. Average precision is the non-interpolated AP metric, rather
than trapezoidal PR AUC. The experiment is internal TCGA-LUSC evaluation and
does not establish performance on an external clinical cohort.

## Frozen protocol and execution

- Source: clean commit `7e8a2b03365978a08693857995f8809f4a5b7df0` on
  `feature/full-cohort-run-20261103`, authored/pushed as
  `mragetsars <mragetsars@gmail.com>`. Source and HEAD remained frozen during
  execution. This curated result is published separately on
  `codex/full-cohort-results-20261104`.
- Python 3.12.14, PyTorch `2.14.0+cu130`, torchvision `0.29.0+cu130`, Windows
  GPU server with NVIDIA GeForce RTX 4070 Ti SUPER. The existing CUDA environment
  was retained and the package installed with `pip install -e ".[dev]"`.
- Seeds `2, 3, 5, 7, 11` × five committed patient-disjoint folds: 25 completed
  independently initialized AMIL fits. Training used Original/Aug1/Aug2 as
  separate bags; validation and test used Original only.
- Maximum 100 epochs, patience 10, `min_delta=0.0001`; Adam learning rate
  `0.0001`, weight decay `0.00001`; batch size 2, attention dimension 128,
  threshold 0.5 and loader workers 0. The committed reference configuration
  had only `device` overridden to `cuda`.
- Checkpoints were selected by validation AUROC, then validation loss. Each
  seed's final model was its best validation fold, with lowest fold index as
  the final tie-break. Test predictions/metrics did not select models, epochs,
  thresholds or settings. This is not a five-fold ensemble or a refit on all
  Train/CV data. See [the experiment protocol](../workflows/experiments.md).

The durable scheduled task `LungWSI-Full-20261103` ran under the existing user's
interactive session with limited privileges and no saved credentials. SSH
disconnect survival was verified. A temporary system-awake request was used;
the power plan was unchanged. The successful controller ran for **4,050.391
seconds**; its training/evaluation command took **2,918.125 seconds**. These
are observed invocation durations, not controlled throughput benchmarks.

## Data, features and completed gates

The 694-slide cohort comprises **373 patients, 347 Normal / 347 Tumor slides**
and **35,366 patches**. Existing patch artifacts from the 2026-10-08
preprocessing/augmentation run were preserved. Before extraction, the current
integrity check hashed **109,199,574,365 raw-WSI bytes** and
**59,966,333,224 saved-HDF5 bytes**; all recorded identities/hashes matched,
with no missing or empty bags. It completed in 102.719 seconds.

Genuine, frozen official KimiaNet weights produced the full cache using CUDA,
batch size 8 and every selected patch/view. It contains **2,082 slide-view bags
and 106,098 embedding vectors of dimension 1024**. The cache was atomically
published after validation; extraction completed in 940.312 seconds. The
full experiment-input validator verified official provenance and all five
patient-split seeds/five folds before fitting.

| Gate/output | Verified result |
| --- | --- |
| General quality | 278 tests passed, 3 expected skips |
| Current-code genuine pilot gate | All 14 slides / 42 full bags passed; 34 tests run, 33 passed, 1 MPS skip; 15 data regressions passed |
| Training | 25/25 fits complete; 579 total epochs |
| Held-out test | 5/5 seeds complete |
| Attention | Raw attention exported for all 699 seed/slide test appearances |
| Spatial outputs | 30 heatmaps generated, six per seed; per-seed and repeated-split ROC figures retained |

The general quality skips were unavailable MPS, an unset real-pilot variable in
the general test phase, and Windows directory-symlink privilege. The explicit
real-pilot gate then passed separately. A pytest-cache permission warning was
nonblocking. All stages of the successful controller exited 0.

## Artifact identity and final audit

All full datasets, model checkpoints, per-slide predictions, attention, figures
and generated reports remain outside Git under:

```text
D:\Meraj Rastegar\Data\AMIL_Full_20261103\run-7e8a2b0
```

Key retained outputs are `full694-kimianet.h5`, `controllerstatus.json`,
`cohort-integrity.json`, `pilot14-acceptance.json`, and
`experiment-reference/{run.json,training/summary.json,test/summary.json}`.
ROC figures are under `experiment-reference/test/`, with seed-specific metrics,
confidence intervals, predictions and attention in `test/seed_*/`. This curated
Markdown publishes aggregate results only; raw IDs/predictions are not added to
Git.

| Identity | Recorded hash/fingerprint |
| --- | --- |
| Full cache size | 415,019,574 bytes |
| Full cache SHA256 | `5f05e126ec60a06114e576b27dfcba7be209061f211b9c6c52a6851b729913e5` |
| Feature fingerprint | `37fedcdfe5ea36692d11f287f799deca6b6b382db3f76cee4ef81b861a8eb61b` |
| Experiment fingerprint | `e7d884b046e029f80cca0e4ddc7b869d81d7726a08c0a0264ed477fea10d2cd2` |
| Reference-config file SHA256 | `7bcaaebe7a9d773a73566d794a9b5f16c55121fb8f38b2f27a76e8003eed058d` |
| Final test-summary SHA256 | `98068d9f67ee4c1f18a162a9553e8c0ddb5ffdc5f0e7c93df6b1b87a23c136f5` |

The retained `final-audit-20261104.json` reports **PASS** at
`2026-11-03T21:25:14.847126Z` (2026-11-04 00:55:14 Asia/Tehran). It verified
**32 sealed stages**, all **61 frozen input snapshots**, the matching code
hashes/clean execution commit, unchanged full feature-cache SHA256, 25 completed
fits, five completed test seeds and the final summary hash. This was a
post-completion checksum verification, not another training/test-inference run.

A separate `independent-results-audit.json` also reports **PASS**, with no
training or inference rerun. It recomputed all five AUROCs from saved prediction
rows using both rank-based and positive/negative pair-comparison calculations,
without importing project metrics or scikit-learn. It also reproduced all
threshold metrics, ten aggregate means/sample SDs and every seed's 1,000-draw
patient-cluster bootstrap interval. Independent checks confirmed all 61 frozen
input hashes, 694 cohort identities, patient separation across the 25 folds and
their held-out tests, Original-only validation/test predictions, and all
validation-only epoch/fold selections. The audit SHA256 is
`7636794dae8e133bfdf29785965a53cee024c5149a5c22a938d7ebbc38a16622`.

The original failed attempt is retained in the parent evidence directory.
It exposed a direct-launch import-path defect after genuine 14/42 checks passed;
the minimal startup fix and CLI regression were committed in `7e8a2b0`, and the
whole chain reran in a fresh output root. Historical source/cache artifacts were
not overwritten. Server synchronization preserved the prior `214ec7e` history
in backup branches and `server-before-sync.bundle`; no private key/token was
copied to the server.
