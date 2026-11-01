# Repository organization review — 2026-11-01

## Findings and changes

| Finding | Resolution |
| --- | --- |
| Main lagged behind accepted feature work; training remained unmerged | Integrated validation/full-cohort work and karmooPA's training on main |
| Alias authors and GitHub committers violated the requested two-name policy | Rewrote actual commit identities with evidence-based ownership mapping |
| Commit subjects used inconsistent styles | Assigned a specific Conventional Commit subject to each original change |
| Stale/duplicate branches obscured active development | One main branch; nine annotated archive milestone tags |
| A historical IDE problem file had been committed | Removed only that file from affected historical trees |
| `src` acted as an ad-hoc import namespace | Standard installable `src/lung_wsi_mil` package and declared build metadata |
| Pipeline implementations lived in scripts imported by tests | Moved implementation into audit/workflows/data/features/augmentation modules |
| CLIs depended on repeated `sys.path` injection | Installed entry points and thin compatibility launchers |
| Requirements and install paths were scattered | Pinned dependency components composed through package extras |
| Plain pytest discovered duplicate scratch test files | Explicit test discovery and removal of scratch/cache folders from the delivery |
| No automated style or commit convention checks | Ruff configuration, Git identity/subject policy, local hook and CI |
| Training counted repeated slide views across batches as new slides | Count distinct slide IDs throughout the epoch |
| Generic `cuda` did not match indexed model devices | Resolve an unspecified device index to the model's device |
| Guides mixed current procedures and historical outcomes | Grouped current guides by topic; preserved dated results and source hashes |
| The requested force-push needed a concrete remote plan | Recorded leases, atomic dry-run/apply helper, backup bundle and recovery steps |

The original test baseline was 161 passed / 3 skipped on the accepted validation
checkout. Integrating the previously unmerged training work produced 230 passed /
4 skipped. Final execution evidence is included in the handoff's quality reports.
The full preprocessing pilot, genuine KimiaNet cache and accelerator tests cannot
be rerun without their external data/weights/devices; their dated evidence is
retained and current optional skips remain explicit.

## Preserved research state

The frozen metadata source bytes, committed cohort metadata, 60 reference split
CSVs and compact historical acceptance JSON are preserved. Model architecture,
patch size/scale, tissue thresholds, augmentation rules, three-view order and
HDF5/Dataset/collate output contracts are unchanged. Code movement/formatting
changes provenance fingerprints. No historical result is relabelled as a new run.

The source MIT license is retained unchanged. Upstream checkpoint terms remain
linked in the acceptance procedure. Empty future scaffolds, virtual environments,
bytecode, test caches, copied server-source trees, session refs and SSH settings
are excluded from the active project. The original ZIP remains their full backup.

## Project work remaining at the organization snapshot

The experiment protocol still needs explicit validation/test view policy,
per-slide aggregation, model selection and evaluation metrics. Full-cohort
feature extraction, trained checkpoints, ROC/AUC and attention heatmaps are
not delivered by this repository cleanup. See the architecture and training guides.

Update: the subsequent [roadmap extension](roadmap-completion.md) implements these experiment/evaluation/heatmap stages. Their real-data execution still requires external inputs.
