"""Exercise the actual identity policy and lease plan without network calls."""

import pytest
from scripts.git_policy import identity_errors, subject_errors
from scripts.push_reorganized import push_command


@pytest.mark.parametrize(
    ("name", "email"),
    [("mragetsars", "mragetsars@gmail.com"), ("karmooPA", "kardostmohammad.h.p@gmail.com")],
)
def test_canonical_identity_pairs_are_accepted(name, email):
    assert identity_errors(name, email, "Author") == []


@pytest.mark.parametrize(
    ("name", "email"),
    [
        ("GitHub", "noreply@github.com"),
        ("Karmo", "kardostmohammad.h.p@gmail.com"),
        ("mragetsars", "kardostmohammad.h.p@gmail.com"),
    ],
)
def test_aliases_and_mismatched_ownership_are_rejected(name, email):
    assert identity_errors(name, email, "Committer")


def test_empty_or_ambiguous_subjects_are_rejected():
    assert subject_errors("")
    assert subject_errors("Update stuff")
    assert not subject_errors("fix(mil): count distinct slides per epoch")


def test_remote_migration_uses_explicit_leases_for_updates_and_deletions():
    plan = {
        "original_branch_tips": {"main": "a" * 40, "feature/old": "b" * 40},
        "archive_tags": ["archive/old"],
    }
    command = push_command(plan, dry_run=True)
    assert "--atomic" in command and "--dry-run" in command
    assert f"--force-with-lease=refs/heads/main:{'a' * 40}" in command
    assert f"--force-with-lease=refs/heads/feature/old:{'b' * 40}" in command
    assert "--force-with-lease=refs/tags/archive/old:" in command
    assert ":refs/heads/feature/old" in command
    assert "main:refs/heads/main" in command
    assert ":refs/heads/main" not in command
    assert "--force" not in command and "--mirror" not in command
