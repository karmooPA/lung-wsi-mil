# Supplementary tests of all 25 frozen AMIL models — 2026-11-04

**COMPLETE: all five seeds × all five folds evaluated**, with no training,
refitting or new model selection. The supplementary mean test AUROC across all
25 correlated models is **0.979433 ± 0.015957** (mean ± sample SD).
The original five validation-selected models remain the primary experiment:
their mean AUROC is **0.974028 ± 0.021111**,
unchanged. See [the primary result](full-cohort-run-2026-11-03.md).

## All 25 results

Every frozen fold's `best.pt` was chosen from validation before this supplementary
test. Each model uses its seed's original held-out patients, Original views only,
Tumor=1 and threshold 0.5. `Primary=Yes` identifies the five models selected in
the original protocol, not a model chosen from these new test scores.

| Seed | Fold | Primary | Best epoch | Validation AUROC | Test AUROC | Test accuracy | Test F1 | Test sensitivity | Test specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | — | 24 | 0.977679 | 0.995573 | 0.985816 | 0.986111 | 1.000000 | 0.971429 |
| 2 | 2 | — | 4 | 0.992857 | 0.989940 | 0.921986 | 0.926174 | 0.971831 | 0.871429 |
| 2 | 3 | Yes | 19 | 0.995372 | 0.995976 | 0.978723 | 0.979021 | 0.985915 | 0.971429 |
| 2 | 4 | — | 9 | 0.989421 | 0.998592 | 0.985816 | 0.985915 | 0.985915 | 0.985714 |
| 2 | 5 | — | 16 | 0.993058 | 0.986519 | 0.985816 | 0.986111 | 1.000000 | 0.971429 |
| 3 | 1 | — | 8 | 0.984694 | 0.975569 | 0.942446 | 0.942857 | 0.942857 | 0.942029 |
| 3 | 2 | Yes | 4 | 0.998677 | 0.985300 | 0.949640 | 0.948905 | 0.928571 | 0.971014 |
| 3 | 3 | — | 7 | 0.990741 | 0.987371 | 0.956835 | 0.956522 | 0.942857 | 0.971014 |
| 3 | 4 | — | 10 | 0.992982 | 0.989234 | 0.949640 | 0.948905 | 0.928571 | 0.971014 |
| 3 | 5 | — | 10 | 0.984091 | 0.994824 | 0.956835 | 0.957143 | 0.957143 | 0.956522 |
| 5 | 1 | Yes | 12 | 1.000000 | 0.975425 | 0.949275 | 0.948905 | 0.942029 | 0.956522 |
| 5 | 2 | — | 15 | 0.993304 | 0.987608 | 0.971014 | 0.970588 | 0.956522 | 0.985507 |
| 5 | 3 | — | 21 | 0.985646 | 0.989498 | 0.963768 | 0.962963 | 0.942029 | 0.985507 |
| 5 | 4 | — | 3 | 0.981818 | 0.991388 | 0.949275 | 0.949640 | 0.956522 | 0.942029 |
| 5 | 5 | — | 16 | 0.982804 | 0.978786 | 0.963768 | 0.963504 | 0.956522 | 0.971014 |
| 7 | 1 | — | 15 | 0.990260 | 0.979074 | 0.929078 | 0.926471 | 0.900000 | 0.957746 |
| 7 | 2 | — | 12 | 0.993506 | 0.977666 | 0.943262 | 0.940299 | 0.900000 | 0.985915 |
| 7 | 3 | — | 6 | 0.976299 | 0.993763 | 0.971631 | 0.971014 | 0.957143 | 0.985915 |
| 7 | 4 | — | 11 | 0.968831 | 0.979678 | 0.929078 | 0.929577 | 0.942857 | 0.915493 |
| 7 | 5 | Yes | 24 | 1.000000 | 0.973642 | 0.943262 | 0.939394 | 0.885714 | 1.000000 |
| 11 | 1 | Yes | 39 | 1.000000 | 0.939796 | 0.914286 | 0.913043 | 0.900000 | 0.928571 |
| 11 | 2 | — | 8 | 0.998017 | 0.961837 | 0.928571 | 0.926471 | 0.900000 | 0.957143 |
| 11 | 3 | — | 9 | 0.996693 | 0.949388 | 0.900000 | 0.898551 | 0.885714 | 0.914286 |
| 11 | 4 | — | 12 | 1.000000 | 0.958367 | 0.928571 | 0.925373 | 0.885714 | 0.971429 |
| 11 | 5 | — | 15 | 0.985390 | 0.951020 | 0.900000 | 0.898551 | 0.885714 | 0.914286 |

## Confidence intervals and remaining test metrics

AUROC intervals are 95% patient-cluster percentile bootstrap intervals, using
1,000 draws per model and the original seed's bootstrap RNG. They describe
slide-level AUROC while resampling patients together. AP is non-interpolated
average precision; confusion counts are ordered TN/FP/FN/TP.

| Seed | Fold | Test AUROC 95% CI | Balanced accuracy | AP | Precision | NPV | Brier | Log loss | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | [0.986530, 1.000000] | 0.985714 | 0.995306 | 0.972603 | 1.000000 | 0.017865 | 0.071185 | 68/2/0/71 |
| 2 | 2 | [0.976631, 0.998295] | 0.921630 | 0.990718 | 0.884615 | 0.968254 | 0.053205 | 0.186949 | 61/9/2/69 |
| 2 | 3 | [0.987206, 1.000000] | 0.978672 | 0.996760 | 0.972222 | 0.985507 | 0.020254 | 0.071944 | 68/2/1/70 |
| 2 | 4 | [0.994707, 1.000000] | 0.985815 | 0.998570 | 0.985915 | 0.985714 | 0.015322 | 0.068681 | 69/1/1/70 |
| 2 | 5 | [0.955882, 1.000000] | 0.985714 | 0.965855 | 0.972603 | 1.000000 | 0.014673 | 0.096891 | 68/2/0/71 |
| 3 | 1 | [0.954602, 0.992877] | 0.942443 | 0.975883 | 0.942857 | 0.942029 | 0.055528 | 0.195025 | 65/4/4/66 |
| 3 | 2 | [0.968884, 0.998018] | 0.949793 | 0.986814 | 0.970149 | 0.930556 | 0.044791 | 0.163663 | 67/2/5/65 |
| 3 | 3 | [0.972416, 0.998522] | 0.956936 | 0.989619 | 0.970588 | 0.943662 | 0.039441 | 0.140454 | 67/2/4/66 |
| 3 | 4 | [0.976910, 0.998289] | 0.949793 | 0.990189 | 0.970149 | 0.930556 | 0.040033 | 0.138457 | 67/2/5/65 |
| 3 | 5 | [0.987143, 0.999397] | 0.956832 | 0.995207 | 0.957143 | 0.956522 | 0.031049 | 0.100479 | 66/3/3/67 |
| 5 | 1 | [0.948700, 0.995306] | 0.949275 | 0.980031 | 0.955882 | 0.942857 | 0.043584 | 0.192125 | 66/3/4/65 |
| 5 | 2 | [0.972056, 0.998994] | 0.971014 | 0.989985 | 0.985075 | 0.957746 | 0.029221 | 0.137012 | 68/1/3/66 |
| 5 | 3 | [0.975121, 0.999541] | 0.963768 | 0.991768 | 0.984848 | 0.944444 | 0.030640 | 0.157045 | 68/1/4/65 |
| 5 | 4 | [0.979844, 0.998956] | 0.949275 | 0.992238 | 0.942857 | 0.955882 | 0.044834 | 0.172773 | 65/4/3/66 |
| 5 | 5 | [0.947330, 1.000000] | 0.963768 | 0.986905 | 0.970588 | 0.957143 | 0.027864 | 0.174186 | 67/2/3/66 |
| 7 | 1 | [0.954458, 0.995595] | 0.928873 | 0.981654 | 0.954545 | 0.906667 | 0.050798 | 0.187897 | 68/3/7/63 |
| 7 | 2 | [0.951183, 0.995196] | 0.942958 | 0.982660 | 0.984375 | 0.909091 | 0.049618 | 0.182985 | 70/1/7/63 |
| 7 | 3 | [0.982449, 1.000000] | 0.971529 | 0.994592 | 0.985294 | 0.958904 | 0.027849 | 0.113561 | 70/1/3/67 |
| 7 | 4 | [0.955210, 0.995768] | 0.929175 | 0.984036 | 0.916667 | 0.942029 | 0.051920 | 0.186035 | 65/6/4/66 |
| 7 | 5 | [0.941776, 0.996067] | 0.942857 | 0.981350 | 1.000000 | 0.898734 | 0.053218 | 0.264627 | 71/0/8/62 |
| 11 | 1 | [0.896007, 0.976308] | 0.914286 | 0.939800 | 0.926471 | 0.902778 | 0.078633 | 0.653231 | 65/5/7/63 |
| 11 | 2 | [0.926494, 0.989374] | 0.928571 | 0.958911 | 0.954545 | 0.905405 | 0.062649 | 0.250250 | 67/3/7/63 |
| 11 | 3 | [0.914902, 0.977008] | 0.900000 | 0.957666 | 0.911765 | 0.888889 | 0.084816 | 0.338726 | 64/6/8/62 |
| 11 | 4 | [0.920824, 0.987857] | 0.928571 | 0.945021 | 0.968750 | 0.894737 | 0.063677 | 0.292146 | 68/2/8/62 |
| 11 | 5 | [0.912973, 0.981508] | 0.900000 | 0.946772 | 0.911765 | 0.888889 | 0.080890 | 0.367339 | 64/6/8/62 |

## Descriptive summaries

The five models within a seed share the same holdout; seed holdouts also overlap.
There are 3,495 model/slide prediction rows but only 466 unique held-out slides.
Means and sample SDs below describe these correlated models, not independent
cohorts, confidence intervals for a population mean, or pooled test AUC. No
highest-test-score fold replaces the original validation-selected model.

| Seed | Five-fold test AUROC mean ± sample SD | Mean test accuracy | Mean test F1 | Original selected fold | Original primary test AUROC |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.993320 ± 0.004940 | 0.971631 | 0.972667 | 3 | 0.995976 |
| 3 | 0.986460 ± 0.007043 | 0.951079 | 0.950866 | 2 | 0.985300 |
| 5 | 0.984541 ± 0.007019 | 0.959420 | 0.959120 | 1 | 0.975425 |
| 7 | 0.980765 ± 0.007637 | 0.943262 | 0.941351 | 5 | 0.973642 |
| 11 | 0.952082 ± 0.008573 | 0.914286 | 0.912398 | 1 | 0.939796 |

| Test metric | All 25 mean | Sample SD |
| --- | --- | --- |
| AUROC | 0.979433 | 0.015957 |
| Average precision | 0.979932 | 0.017356 |
| Accuracy | 0.947936 | 0.024960 |
| Balanced accuracy | 0.947891 | 0.024968 |
| Sensitivity | 0.937606 | 0.036614 |
| Specificity | 0.958175 | 0.029935 |
| Precision | 0.958091 | 0.028610 |
| NPV | 0.939880 | 0.033852 |
| F1 | 0.947280 | 0.025400 |
| Brier score | 0.044495 | 0.019845 |
| Log loss | 0.196147 | 0.122898 |

## Provenance and exported detail

The standalone evaluator used the frozen code/config/runtime/checkpoints and
original feature cache. Before and after inference it verified the original
32 completed stages, frozen inputs, source code and feature identity. All five
original selected models reproduced every saved prediction column, probability
bits, metrics, bootstrap interval and test loss exactly. The original experiment
and its model selections were not modified.

An independent saved-output audit reports PASS for all 25 models, including
rank-based and pairwise AUROC calculations, threshold metrics, bootstrap
intervals, frozen holdout identities, checkpoint provenance, CV values and the
complete 579-epoch history. This audit performed no training or inference. It verified 26 sealed stages, 2,771 validation prediction rows, 3,495 test prediction rows and all 25,000 bootstrap draws.

| Identity | Value |
| --- | --- |
| Frozen execution code | `7e8a2b03365978a08693857995f8809f4a5b7df0` |
| Original run fingerprint | `e7d884b046e029f80cca0e4ddc7b869d81d7726a08c0a0264ed477fea10d2cd2` |
| Supplementary evaluation fingerprint | `4c54dca09ae81f50330ca5ef4a01f85317070c9ff303636e06623d8ba1395235` |
| Standalone evaluator SHA256 | `63ca545f25bce5f155be00cf930cc829d95161b5aee8bbcae2ac7b8ead6eba26` |
| Full feature-cache SHA256 | `5f05e126ec60a06114e576b27dfcba7be209061f211b9c6c52a6851b729913e5` |
| Supplementary summary SHA256 | `4cffb5154b0349aeab9eee3b5ef5310cf2a5c780ee76bcedaf25dfbc3d13c085` |
| Independent audit SHA256 | `0232b5d8ead54a87dd57a54c0cd2b17363082fb556220b16c6626c9a33d38171` |
| Sealed ZIP SHA256 | `29efb6b65571388dbe36bd00529a0335ed458dd4d001f80b071a1e62367e58b3` |
| Sealed ZIP size / files | 10,580,546 bytes / 7,278 files |

All generated artifacts remain outside Git at:

```text
D:\Meraj Rastegar\Data\AMIL_Full_20261103\all-fold-tests-20261104
```

`all_models.csv` and `all_models.json` contain the complete 25-row validation/test
metrics, selected epochs, original primary flags, checkpoint hashes and
confidence intervals. `epoch_history.csv`/`.json` contain all **579 original
training epochs**, including train loss and validation loss/AUROC. No new epoch
was trained. `by_seed_summary.json` retains per-seed descriptive summaries.
Each `seed_N/fold_F/` contains predictions, metrics, curves, bootstrap results,
ROC/PR/confusion figures, copied CV metrics/history and patch-aligned attention.
The sealed sibling ZIP includes these files and the evaluator script.

Original training and primary-test evidence remains at:

```text
D:\Meraj Rastegar\Data\AMIL_Full_20261103\run-7e8a2b0\experiment-reference
```

Only this curated aggregate report is tracked; generated CSV/JSON, raw slide
identifiers/predictions, figures, attention and archives are not committed.
