# Documentation

## Design and development

- [Architecture and data contracts](architecture.md)
- [Workspace and external data](development/workspace.md)
- [Contributor ownership and attribution](development/ownership.md)
- [Pipeline stages and Git history](development/pipeline-history.md)
- [History rewrite, backup, and remote migration](development/git-history.md)
- [Contribution workflow](../CONTRIBUTING.md)
- [Organization review](development/organization-review.md)

## Data and execution

- [One-command pipeline and external Data layout](workflows/pipeline.md)
- [Dataset audit and provenance](data/dataset-audit.md)
- [Patch augmentation](data/augmentation.md)
- [Preprocessing](workflows/preprocessing.md)
- [Frozen 14-slide pilot](workflows/pilot14.md)
- [Full-cohort processing](workflows/full-dataset.md)
- [KimiaNet / AMIL real-data acceptance](workflows/amil-validation.md)
- [Batch and epoch training](workflows/training.md)
- [Complete 5×5 experiment and heatmaps](workflows/experiments.md)
- [Roadmap implementation status](development/roadmap-completion.md)

## Recorded results

- [2026-11-04 supplementary tests of all 25 frozen checkpoints](results/all-fold-tests-2026-11-04.md)
- [2026-11-03 full-cohort AMIL training and primary held-out results](results/full-cohort-run-2026-11-03.md)
- [2026-11-01 roadmap software validation (synthetic only)](results/roadmap-software-validation-2026-11-01.md)
- [Machine-readable roadmap software validation](results/roadmap-software-validation-2026-11-01.json)
- [2026-10-02 preprocessing pilot](results/pilot14-2026-10-02.md)
- [2026-10-17 KimiaNet / AMIL acceptance](results/amil-validation-2026-10-17.md)
- [Machine-readable acceptance](results/amil-validation-2026-10-17.json)

Dated reports retain their original result counts, source/artifact hashes,
commit IDs and limits. Their external paths, commands and branch names describe
those runs. On the project server, compatibility aliases keep recorded paths
resolvable after the Data migration. Use the [current pipeline layout](workflows/pipeline.md)
for new work. The commit map explains rewritten Git IDs without relabelling an
old run as a new one.
