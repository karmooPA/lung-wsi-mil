#!/usr/bin/env python3
"""Check commit identities and Conventional Commit subjects without dependencies."""

import argparse
import os
import re
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
IDENTITIES = {
    "mragetsars": "mragetsars@gmail.com",
    "karmooPA": "kardostmohammad.h.p@gmail.com",
}
SUBJECT = re.compile(
    r"(?:build|chore|ci|docs|feat|fix|merge|perf|refactor|revert|style|test)"
    r"(?:\([a-z0-9][a-z0-9-]*\))?!?: \S.*"
)


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT, capture_output=True, text=True, check=True
    ).stdout.strip()


def identity_errors(name, email, role):
    if IDENTITIES.get(name) != email:
        return [f"{role} must use one of the two documented name/email pairs: {name} <{email}>"]
    return []


def subject_errors(subject):
    return (
        [] if SUBJECT.fullmatch(subject) else [f"Invalid Conventional Commit subject: {subject!r}"]
    )


def check_history(revision="HEAD"):
    """Inspect raw identities, rather than hiding aliases through .mailmap."""
    errors = []
    rows = git("log", revision, "--format=%H%x09%an%x09%ae%x09%cn%x09%ce%x09%s").splitlines()
    for row in rows:
        commit, author, author_email, committer, committer_email, subject = row.split("\t", 5)
        issues = [
            *identity_errors(author, author_email, "Author"),
            *identity_errors(committer, committer_email, "Committer"),
            *subject_errors(subject),
        ]
        errors.extend(f"{commit[:12]}: {issue}" for issue in issues)
    return len(rows), errors


def check_pending_commit(message_path):
    errors = []
    for role in ("AUTHOR", "COMMITTER"):
        ident = git("var", f"GIT_{role}_IDENT")
        match = re.fullmatch(r"(.*?) <(.*?)> \d+ [+-]\d{4}", ident)
        if match is None:
            errors.append(f"Cannot parse {role.lower()} identity")
        else:
            errors.extend(identity_errors(*match.groups(), role.capitalize()))
    lines = Path(message_path).read_text(encoding="utf-8").splitlines()
    errors.extend(subject_errors(lines[0] if lines else ""))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=os.environ.get("LUNG_WSI_HISTORY_REVISION", "HEAD"))
    parser.add_argument(
        "--commit-message", type=Path, help="Validate a pending commit in commit-msg."
    )
    args = parser.parse_args(argv)
    try:
        if args.commit_message:
            count, errors = 1, check_pending_commit(args.commit_message)
        else:
            count, errors = check_history(args.revision)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Git policy could not run: {error}\n")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"Git policy passed: {count} commit(s), two canonical identities, consistent subjects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
