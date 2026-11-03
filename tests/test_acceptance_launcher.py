"""Exercise the real acceptance entry point without pytest's import-path setup."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_direct_acceptance_launcher_reports_only_missing_real_pilot(tmp_path):
    project = Path(__file__).resolve().parents[1]
    report_path = (tmp_path / "acceptance.json").resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PILOT14_H5", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/validate_amil.py"),
            "--output",
            str(report_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "ModuleNotFoundError" not in output, output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["amil_tests"]["errors"] == 0, report["amil_tests"]
    assert report["amil_tests"]["failures"] == 0, report["amil_tests"]
    assert report["suite_coverage"]["status"] == "pass", report["suite_coverage"]
    assert report["data_regressions"]["status"] == "pass", report["data_regressions"]
    assert report["data_regressions"]["counts"].get("passed", 0) > 0
    required_skips = [
        item
        for item in report["amil_tests"]["outcomes"]
        if item["status"] == "skip" and not item["optional_device"]
    ]
    assert len(required_skips) == 1, required_skips
    assert required_skips[0]["test"] == (
        "test_amil_pipeline.AMILPilot14Tests.test_real_pilot14_all_42_full_bags"
    )
    assert "PILOT14_H5 is unset" in required_skips[0]["reason"]
    assert report["pilot14"]["status"] == "not_run"
    assert report["pilot14_file"]["error"] == "PILOT14_H5 is unset"
    assert not report["acceptance"]["real_pilot14_14_wsi_42_full_forwards"]
