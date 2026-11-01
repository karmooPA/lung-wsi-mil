# Rewritten history and remote migration

## Delivered state

The repository is on integrated `main`. All 28 original reachable commits remain
represented with their parent order, timestamps and contributor ownership.
Author and committer names/emails are canonicalized, commit subjects follow
Conventional Commits, and the accidentally tracked `.cph` IDE problem file is
removed from historical trees. No scientific implementation is dropped.

The newest AMIL training commit was present only on the supplied remote-tracking
`feature/amil-implementation` ref. It is now merged with the accepted pipeline.
Nine annotated `archive/*` tags retain each supplied remote branch milestone.
The duplicate `codex/pilot14-hardening` branch pointed to the same commit as the
old main and needs no second archive. Session-specific `refs/codex/*`, reflogs,
cache files and machine-specific remote/editor settings are not carried forward.

The handoff contains `original-history.bundle` outside the active checkout. It
preserves the untouched original branch history, including original hosting
signatures and aliases. The active repository uses only the two required names.
The original supplied ZIP is the complete backup of untracked scratch contents.

See [commit-map.csv](commit-map.csv) for old → new IDs and
[remote-plan.json](remote-plan.json) for the exact last-observed remote tips.
Dated evidence documents retain old IDs and original result claims. Rewritten
commits no longer have their old GitHub signatures; hashes/signatures are not
presented as if unchanged.

## Configure attribution

The delivered local identity matches the supplied checkout: `mragetsars`.
Select `karmooPA` before committing that contributor's work:

```bash
python scripts/configure_git.py karmooPA
python scripts/git_policy.py
git status
```

Only repository-local configuration is changed. No global settings, SSH keys,
tokens, credentials, dataset paths or server SSH configuration are copied.
`origin` remains `git@github.com:karmooPA/lung-wsi-mil.git`.

## Review and publish

### Corrected roadmap attribution

The 5-Fold training foundation now uses `karmooPA` for both author and committer;
the remaining roadmap work retains `mragetsars`. A mixed implementation commit
was split without changing implementation bytes. Its descendants have new IDs;
the [attribution map](roadmap-attribution-correction.json) records the relation.
The handoff's `roadmap-before-attribution-correction.bundle` preserves the exact
previously delivered graph and the original IDs referenced in validation evidence.

If remote main is already the **previous roadmap delivery**
`b0983878c3d75aeb5396eccefc85f1eb61c1e81b`, publish this correction with its exact
lease after reviewing the new history:

```bash
git push --force-with-lease=refs/heads/main:b0983878c3d75aeb5396eccefc85f1eb61c1e81b origin main:refs/heads/main
```

The lease rejects a remote that has advanced since that delivery. Review any
intervening work rather than weakening the lease. No remote was contacted or
pushed while preparing this correction.

### Earlier organized or original remote

If the organized base `689763dbed617cf8daf7d4c29b6ff22f6d7f1d39`
has already been published and remote main has no divergent changes, this
roadmap extension is a normal fast-forward:

```bash
git push origin main
```

The migration helper below is for a remote that still has the original
pre-organization tips from the supplied ZIP. Its original leases deliberately
remain unchanged. Do not rerun that migration against an already migrated
remote or refresh its leases without reviewing intervening work.

Extract the whole repository, including `.git`; do not create a new Git repository
or copy only visible source files. From the delivered `lung-wsi-mil/` directory:

```bash
git status
git log --graph --oneline --decorate --all
python scripts/git_policy.py
python scripts/push_reorganized.py
```

The last command only prints the migration plan. It does not contact a remote.
It targets the recorded repository, checks a clean main checkout, and checks
that every archived branch milestone is an ancestor of integrated main.

When you are ready to update GitHub yourself:

```bash
python scripts/push_reorganized.py --dry-run
python scripts/push_reorganized.py --apply
```

`--apply` runs a single atomic push. It replaces main, creates the nine archive
tags and deletes the eight completed feature branches in the recorded plan.
Every replacement/deletion uses its exact original SHA as a lease; every new
archive tag has an empty-target lease. An unknown branch is not deleted.
If any tip changed after the ZIP snapshot, the lease rejects the operation.
Review that new work before updating the plan. Branch protection or a tag name
conflict may also reject the push; atomic mode leaves the ref update unapplied.

No fetch is used to weaken the recorded leases, no mirror push is used, and no
remote push has been executed during this organization. Coordinate the history
replacement with the other contributor; their old clones will refer to old IDs.

If you want to replace only main and retain remote feature branches for now,
use this narrower alternative instead of the full-plan apply:

```bash
git push --force-with-lease=refs/heads/main:744df59d6f2d361cf60de03564576ff65132b1dd origin main:refs/heads/main
git push origin --tags
```

Those retained remote feature branches still expose their old history until
you separately archive/remove them. The full-plan helper completes both tasks.

## Recovery

The original bundle is deliberately separate from the reorganized repository:

```bash
git clone ../handoff/original-history.bundle ../lung-wsi-mil-before-organization
```

This creates a separate checkout; it does not alter the cleaned one. Use the
bundle's refs and original-tip JSON when investigating an old commit/report.
