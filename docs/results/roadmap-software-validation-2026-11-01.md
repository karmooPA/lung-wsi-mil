# Roadmap software validation — 2026-11-01

**Scope: synthetic software verification only. No new real-data training or
diagnostic performance is claimed.** Following the user's attribution correction,
the 5-Fold training foundation is authored and committed by `karmooPA`;
the remaining roadmap integration and independent verification use `mragetsars`.

The tested clean source commit is
`e9cc86fcf2be2617e5e23d07c658790bae515904`, package version `0.2.0`.
This report was added after that run. Subsequent attribution/documentation
corrections preserve the tested implementation. The original tested ID remains
in the dated evidence and is recoverable from the handoff's pre-correction
bundle; the [attribution map](../development/roadmap-attribution-correction.json)
relates it to the corrected graph. The [machine-readable record](roadmap-software-validation-2026-11-01.json)
binds the exact implementation hashes, run fingerprint, config, runtime and
selected output checksums.

| Check | Observed result |
| --- | --- |
| Full unit/integration suite | 276 passed, 4 skipped |
| Complete quality runner | All 10 checks passed |
| Package builds | Wheel and source distribution succeeded |
| Independent wheel installation | All 12 console entrypoints loaded and returned help successfully outside the checkout |
| Installed implementation | All 30 experiment-related source hashes matched the checkout |
| Public `smoke` CLI | Exit code 0; 30 synthetic patients, 60 synthetic slides |
| Training | All five seeds × five folds completed; 25 fits, two epochs each |
| Final testing | Five validation-selected models; 60 slide/seed predictions |
| Attention | 60 raw patch-coordinate/weight tables |
| Spatial outputs | 10 synthetic heatmaps and overlays |
| Completed artifact integrity | Training/test file manifests and checksums verified |
| Visual review | Synthetic metrics and attention overview figures inspected |

The four skips are two unavailable CUDA checks, one unavailable MPS check and
the external `PILOT14_H5` real-data acceptance gate. The existing dated real-data
reports are preserved, but their experiments were not rerun.

The suite also verifies interrupted/resumed CPU training against uninterrupted
weights and epoch histories, Original-only inference, isolation of test data
from fitting and selection, patient grouping, corrupted input/checkpoint
rejection, undefined metric reporting and coordinate-aligned attention maps.

Reproduce the software run from the checkout:

```bash
python -m pip install -e ".[dev]"
lung-wsi-experiment smoke --output ../lung-wsi-mil-outputs/software-smoke
```

The output folder must be new. The smoke data intentionally has a learnable
synthetic signal; its numerical AUC is not a TCGA-LUSC result and is omitted
from this acceptance summary. Labelled sample figures, raw verification logs
and compact reports accompany the ZIP in `handoff/verification/` and
`handoff/software-smoke-example/`. Synthetic caches and model checkpoints are
not required to use the delivered repository.

For real execution, supply the complete external KimiaNet feature cache and
source preprocessing geometry/QC thumbnails, then follow the
[experiment guide](../workflows/experiments.md). CPU software checks do not
establish execution results on unavailable CUDA/MPS hardware.
