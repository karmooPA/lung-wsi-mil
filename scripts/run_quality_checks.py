#!/usr/bin/env python3
"""Run the repository's reproducible compile, unit, audit, and pilot checks."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT.parent / f"{PROJECT.name}-outputs"


def _run(name, command, env):
    print(f"\n=== {name} ===", flush=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "status": "PASSED" if completed.returncode == 0 else "FAILED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--verify-source-md5", action="store_true")
    parser.add_argument("--source-sample-count", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "quality",
        help="Directory for all quality reports (outside the repository by default).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional override for the summary JSON path.",
    )
    args = parser.parse_args()
    if args.source_sample_count < 0:
        parser.error("--source-sample-count must be >= 0")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    python = sys.executable
    commands = [
        ("Compile Python", [python, "-m", "compileall", "-q", "scripts", "src", "tests"]),
        ("Unit and integration tests", [python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]),
        (
            "Frozen source validation",
            [python, "scripts/validate_sources.py", "--output", str(output_dir / "source_validation.json")],
        ),
        (
            "Recorded audit replay",
            [python, "scripts/inspect_dataset.py", "--mode", "recorded", "--output-dir", str(output_dir / "audit-recorded")],
        ),
        ("Installed dependency consistency", [python, "-m", "pip", "check"]),
        ("Committed-tree whitespace check", ["git", "show", "--check", "--format=", "HEAD"]),
        ("Working-tree whitespace check", ["git", "diff", "--check", "HEAD"]),
    ]
    if args.pilot_output:
        verify = [
            python,
            "scripts/verify_pilot14.py",
            str(args.pilot_output),
            "--source-sample-count",
            str(args.source_sample_count),
        ]
        if args.verify_source_md5:
            verify.append("--verify-source-md5")
        commands.append(("Pilot-14 artifact verification", verify))

    results = [_run(name, command, env) for name, command in commands]
    report = {
        "status": "PASSED" if all(item["exit_code"] == 0 for item in results) else "FAILED",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "project": str(PROJECT),
        "checks": results,
    }
    report_path = (args.report or output_dir / "quality_check_summary.json").expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(f"\nOverall: {report['status']}\nReport: {report_path}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
