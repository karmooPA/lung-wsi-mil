#!/usr/bin/env python3
"""Preview or execute the supplied, atomic, lease-protected remote migration."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "docs/development/remote-plan.json"


def push_command(plan, *, dry_run=False):
    """Replace only the recorded branch tips and create the preserved archive tags."""
    command = ["git", "push", "--set-upstream", "--atomic"]
    if dry_run:
        command.append("--dry-run")
    for name, original in sorted(plan["original_branch_tips"].items()):
        command.append(f"--force-with-lease=refs/heads/{name}:{original}")
    for tag in sorted(plan["archive_tags"]):
        command.append(f"--force-with-lease=refs/tags/{tag}:")
    command.extend(["origin", "main:refs/heads/main"])
    command.extend(f"refs/tags/{tag}:refs/tags/{tag}" for tag in sorted(plan["archive_tags"]))
    command.extend(
        f":refs/heads/{name}" for name in sorted(plan["original_branch_tips"]) if name != "main"
    )
    return command


def check_local_plan(plan):
    remote = subprocess.run(
        ["git", "remote", "get-url", "--push", "origin"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote not in plan["allowed_remote_urls"]:
        raise ValueError(f"origin is outside the recorded target repository: {remote}")
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=PROJECT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if branch != "main":
        raise ValueError("Switch to the reorganized main branch first.")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise ValueError("The source working tree must be clean before migration.")
    for tag in plan["archive_tags"]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", f"refs/tags/{tag}^{{commit}}", "main"],
            cwd=PROJECT,
            check=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Ask origin to check the push without updating refs."
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Publish the main/tag update and delete the archived feature branches.",
    )
    args = parser.parse_args(argv)
    try:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        check_local_plan(plan)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Cannot use the recorded migration plan: {error}\n")
    command = push_command(plan, dry_run=args.dry_run)
    print(shlex.join(command))
    if args.apply or args.dry_run:
        return subprocess.run(command, cwd=PROJECT, check=False).returncode
    print(
        "Preview only. No remote was contacted. Read docs/development/git-history.md before --apply."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
