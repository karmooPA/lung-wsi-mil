#!/usr/bin/env python3
"""Select one contributor's repository-local identity and enable commit checks."""

import argparse
import subprocess
from pathlib import Path

IDENTITIES = {
    "mragetsars": "mragetsars@gmail.com",
    "karmooPA": "kardostmohammad.h.p@gmail.com",
}
PROJECT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity", choices=IDENTITIES)
    args = parser.parse_args(argv)
    settings = {
        "user.name": args.identity,
        "user.email": IDENTITIES[args.identity],
        "user.useConfigOnly": "true",
        "core.hooksPath": ".githooks",
        "core.autocrlf": "false",
        "core.safecrlf": "true",
        "fetch.prune": "true",
        "pull.ff": "only",
        "push.default": "simple",
        "merge.conflictStyle": "zdiff3",
    }
    for name, value in settings.items():
        subprocess.run(["git", "config", "--local", name, value], cwd=PROJECT, check=True)
    print(f"Repository identity: {args.identity} <{IDENTITIES[args.identity]}>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
