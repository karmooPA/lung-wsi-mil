# Contributing

## Setup and checks

Use Python 3.12 and `python -m pip install -e ".[dev]"` after installing the
appropriate torch/torchvision pair. Run `python scripts/run_quality_checks.py`
and `python -m build` before submitting changes. Ruff config and pytest discovery
live in `pyproject.toml`; formatting uses 100-column Python code.

## Identity and ownership

Choose the identity of the contributor whose work you are committing:

```bash
python scripts/configure_git.py mragetsars
# Or, for karmooPA's changes:
python scripts/configure_git.py karmooPA
```

This changes only repository-local settings, enables the checked-in commit
hook and selects the [commit template](.gitmessage.txt). Author **and** committer
must use the same canonical pair for each commit.

| Name | Email |
| --- | --- |
| `mragetsars` | `mragetsars@gmail.com` |
| `karmooPA` | `kardostmohammad.h.p@gmail.com` |

Keep commits scoped to one contributor's responsibility. Cross-module callers
may move with an API migration; retain original ancestry and avoid attributing
another person's implementation to the integrator. See the
[ownership audit](docs/development/ownership.md). A name/email setting records
attribution; it does not create a cryptographic signature or establish authorship.

## Branches and commits

`main` is the integrated branch and the checkout to use for current runs. Start
short-lived branches such as `feat/mil-validation`, `fix/pilot-resume`,
`refactor/data-contracts` or `docs/experiment-protocol` from `main`. Retain the
numbered `pipeline/*` branches as fixed references to the integrated stages
listed in the [pipeline history](docs/development/pipeline-history.md).
The `archive/*` tags preserve milestones from the earlier organization.

Use the commit template for a specific Conventional Commit subject. In the
body, record why the change was made and what verified it when useful:

```text
feat(mil): add patient-level validation metrics
fix(workflows): reject stale source identities on resume
docs(data): explain frozen cohort selection
```

Keep commits small enough to review and include the relevant verification in the
PR description. Local hooks check identities and subjects; CI checks actual
history as well. Perform merges locally with one of the two allowed identities
if the policy must also hold for merge committers; hosting-generated merge
commits can introduce a service identity. Original signatures cannot survive
a history rewrite, so no replacement signature is fabricated.

## Files and scientific changes

Reusable implementation belongs under `src/lung_wsi_mil/`; repository-only tools
and compatibility launchers belong under `scripts/`. Update callers, doc links,
entry points and provenance code inputs when moving code. Add tests for changed
behavior and keep the public HDF5/Dataset/collate/AMIL contracts explicit.

Keep raw WSIs, model weights, caches, virtual environments and generated outputs
outside version control. Do not normalize or edit frozen `metadata/source/`
bytes. Preserve dated reports and reference splits; new results need a new run
identity. Never turn a missing-data skip into a real-data PASS.
