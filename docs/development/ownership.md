# Contributor ownership and history evidence

Ownership was determined from the supplied Git graph and per-commit file changes,
not guessed from account names or the branch owner. The complete original graph
is retained in the handoff's `original-history.bundle`; the rewritten graph
preserves both contributors' ancestry, parent order and original timestamps.

| Contributor | Established responsibility | Original evidence commits |
| --- | --- | --- |
| `karmooPA` | Repository foundation, WSI reader, tissue detection, patch extraction and batch preprocessing | `007d742`, `9f83bb4`, `19675ad`, `1e136b3`, `a3f699f` |
| `karmooPA` | KimiaNet encoder and batched feature APIs | `6b66a1c`, `b142bc2` |
| `karmooPA` | MIL Dataset/collate/DataLoader and patient-aware splits/folds | `3d5ad72`, `460a355`, `d4999aa` |
| `karmooPA` | AMIL, initialization, padding removal, BCE and batch/epoch training | `3424388`, `4b2a2d7`, `dc6018d` |
| `mragetsars` | Frozen cohort audit, source provenance and audit tests | `067f5d1` |
| `mragetsars` | Original/Aug1/Aug2 transforms, storage and pilot hardening | `027fdcd`, `744df59` |
| `mragetsars` | Full-cohort orchestration, independent verification and server integration | `cf42b2b`, `9801697` |
| `mragetsars` | Strict KimiaNet checkpoint/cache provenance, AMIL contracts and recorded acceptance | `c4ce4a1`, `214ec7e` |

Some files have contributions from both people. For example, KimiaNet originated
with karmooPA; strict safe checkpoint loading and streaming cache acceptance were
later added by mragetsars. Preprocessing originated with karmooPA; resume and
physical workspace hardening include mragetsars's later changes. Preserving the
history keeps these distinctions visible.

The source AMIL contract excerpt also distinguishes model implementation from
independent test/acceptance work. The test implementation is not reassigned to
the model author during cleanup. New core API moves, core test formatting and
training corrections use karmooPA; audit/workflow/acceptance organization and
repository tooling use mragetsars. The integration merge uses mragetsars while
retaining karmooPA's original training commit as a parent ancestor.

## Alias normalization

| Observed identity | Canonical identity | Evidence |
| --- | --- | --- |
| `Rastegar` with `161657590+mragetsars@users.noreply.github.com` | `mragetsars` | Embedded GitHub username and corresponding project author email |
| `Karmo` with `165961476+karmooPA@users.noreply.github.com` | `karmooPA` | Embedded GitHub username and corresponding project author email |
| `GitHub` as committer of those web commits | Same canonical contributor as that commit's author | Hosting integration identity, not an additional project contributor |

Canonical emails are taken from existing commits. Names and emails are rewritten
on the actual objects, not only hidden by `.mailmap`. Conventional subjects are
specific to each original change. The [commit map](commit-map.csv) records every
old/new object ID and original/new subject. Historical reports intentionally keep
the IDs of the code that actually generated their evidence.

## Roadmap extension — 2026-11-01

The user clarified that the **5-Fold training part belongs to `karmooPA`**.
The previously mixed implementation commit was split at the training-module
boundary, with the following actual author and committer identities:

| Contributor | Roadmap contribution |
| --- | --- |
| `karmooPA` | Fold training, validation-based checkpoint selection, early stopping, exact resume and their shared configuration/artifact/RNG support |
| `mragetsars` | Five-seed orchestration, final held-out evaluation, metrics, attention/heatmaps, independent tests and documentation |

The training foundation is commit
`2dc71e5835d25e8f09b75a4b75585223e74b56c0`; the remaining workflow integration
is `7b8f44843e80a3480fa63f0f882d560ee910b1f3`.
The split preserves every implementation byte, original commit timestamps
and earlier ancestry. The [attribution map](roadmap-attribution-correction.json)
records replacement IDs, exact training paths and the previous-delivery backup.
Original AMIL/training/data contributions retain their canonical authors.
