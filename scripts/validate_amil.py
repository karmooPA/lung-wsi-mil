#!/usr/bin/env python3
"""Run the AMIL contract, genuine Pilot14 gate, and data-layer regressions.

An unset PILOT14_H5 is a useful development skip in unittest, but this final
acceptance command exits nonzero until all 14 real WSI / 42 views pass.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_REGRESSIONS = (
    "tests/test_mil_hdf5_dataset.py",
    "tests/test_mil_collate.py",
    "tests/test_mil_dataloader.py",
    "tests/test_fold_dataloaders.py",
    "tests/test_patient_split.py",
    "tests/test_cross_validation.py",
    "tests/test_saved_splits.py",
)
REQUIRED_TEST_CLASSES = {
    "test_amil.AMILContractTests": 18,
    "test_amil.AMILInitializationTests": 5,
    "test_amil_pipeline.AMILPipelineTests": 8,
    "test_amil_pipeline.AMILPilot14Tests": 1,
}
OPTIONAL_DEVICE_TEST_IDS = {
    "test_amil.AMILContractTests.test_cuda_forward_and_backward_when_available": "cuda",
    "test_amil.AMILContractTests.test_mps_forward_and_backward_when_available": "mps",
}


class RecordingResult(unittest.TextTestResult):
    """Keep named outcomes, including optional accelerator skip reasons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"test": test.id(), "status": "pass"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.outcomes.append(
            {"test": test.id(), "status": "fail", "detail": self._exc_info_to_string(error, test)}
        )

    def addError(self, test, error):
        super().addError(test, error)
        self.outcomes.append(
            {"test": test.id(), "status": "error", "detail": self._exc_info_to_string(error, test)}
        )

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        device = OPTIONAL_DEVICE_TEST_IDS.get(test.id())
        available = (
            torch.cuda.is_available()
            if device == "cuda"
            else bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            if device == "mps"
            else True
        )
        self.outcomes.append(
            {
                "test": test.id(),
                "status": "skip",
                "reason": reason,
                "optional_device": device is not None and not available,
            }
        )

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        if error is not None:
            status = "fail" if issubclass(error[0], test.failureException) else "error"
            self.outcomes.append(
                {
                    "test": subtest.id(),
                    "status": status,
                    "test_case": test.id(),
                    "detail": self._exc_info_to_string(error, test),
                }
            )


def _git(*arguments):
    completed = subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.rstrip("\n")


def _code_identity():
    try:
        identity = {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_porcelain": _git("status", "--porcelain=v1"),
        }
        identity["dirty"] = bool(identity["status_porcelain"])
    except (OSError, RuntimeError) as error:
        identity = {"commit": None, "branch": None, "dirty": None, "error": str(error)}
    # Dirty working trees still have an auditable identity for the exact files
    # under test; generated report files do not enter this fingerprint.
    paths = sorted(
        {
            *PROJECT_ROOT.glob("src/lung_wsi_mil/mil/*.py"),
            *PROJECT_ROOT.glob("src/lung_wsi_mil/data/*.py"),
            *PROJECT_ROOT.glob("tests/test_amil*.py"),
            Path(__file__).resolve(),
        }
    )
    identity["tested_file_sha256"] = {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return identity


def _file_identity(path):
    if not path:
        return {"path": None, "error": "PILOT14_H5 is unset"}
    target = Path(path).expanduser().resolve()
    result = {"path": str(target)}
    try:
        stat = target.stat()
        result.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    except OSError as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _regressions():
    command = [sys.executable, "-m", "pytest", "-q", *DATA_REGRESSIONS]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    output = completed.stdout + completed.stderr
    counts = {}
    for key in ("passed", "failed", "skipped", "errors", "error", "deselected"):
        matches = re.findall(rf"\b(\d+) {key}\b", output)
        if matches:
            counts[key] = int(matches[-1])
    complete = (
        completed.returncode == 0
        and counts.get("passed", 0) > 0
        and all(
            counts.get(key, 0) == 0
            for key in ("failed", "skipped", "errors", "error", "deselected")
        )
    )
    return {
        "status": "pass" if complete else "fail",
        "command": command,
        "returncode": completed.returncode,
        "counts": counts,
        "output": output[-20000:],
    }


def _suite_coverage(suite):
    """Fail closed when discovery silently omits a required module/class."""
    ids = []

    def collect(item):
        if isinstance(item, unittest.TestSuite):
            for child in item:
                collect(child)
        else:
            ids.append(item.id())

    collect(suite)
    counts = {
        class_id: len({test_id for test_id in ids if test_id.rsplit(".", 1)[0] == class_id})
        for class_id in REQUIRED_TEST_CLASSES
    }
    missing = {
        class_id: {"minimum": minimum, "collected": counts[class_id]}
        for class_id, minimum in REQUIRED_TEST_CLASSES.items()
        if counts[class_id] < minimum
    }
    return {
        "status": "fail" if missing else "pass",
        "counts_by_required_class": counts,
        "required_class_minimums": REQUIRED_TEST_CLASSES,
        "missing": missing,
        "collected_test_ids": ids,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot14-h5",
        default=os.environ.get("PILOT14_H5"),
        help="Genuine 14-WSI KimiaNet HDF5; alternatively set PILOT14_H5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "amil-validation.json",
        help="Machine-readable acceptance report",
    )
    args = parser.parse_args(argv)
    if args.pilot14_h5:
        os.environ["PILOT14_H5"] = str(Path(args.pilot14_h5).expanduser().resolve())
    else:
        os.environ.pop("PILOT14_H5", None)
    started = time.monotonic()
    report = {
        "schema_version": "1.0",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "code": _code_identity(),
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "pytorch": torch.__version__,
            "required_device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
        "pilot14_file": _file_identity(args.pilot14_h5),
        "warnings": [
            "Random initialized forward checks do not measure diagnostic "
            "accuracy, ROC/AUC, or trained model performance."
        ],
    }
    suite = unittest.defaultTestLoader.discover(
        str(PROJECT_ROOT / "tests"), pattern="test_amil*.py"
    )
    report["suite_coverage"] = _suite_coverage(suite)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
    report["amil_tests"] = {
        "run": result.testsRun,
        "passed": sum(item["status"] == "pass" for item in result.outcomes),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "outcomes": result.outcomes,
    }
    # unittest counts a failed subtest as an individual failure/error event.
    # Also expose method counts so totals remain interpretable when one method
    # intentionally checks many invalid inputs.
    report["amil_tests"]["failed_test_methods"] = len(
        {
            item.get("test_case", item["test"])
            for item in result.outcomes
            if item["status"] in {"fail", "error"}
        }
    )
    report["device_results"] = {}
    for device in ("cpu", "cuda", "mps"):
        outcomes = [
            item
            for item in result.outcomes
            if (
                device in item["test"].lower()
                if device != "cpu"
                else not any(optional in item["test"].lower() for optional in ("cuda", "mps"))
            )
        ]
        statuses = {item["status"] for item in outcomes}
        status = (
            "fail"
            if statuses & {"fail", "error"}
            else "incomplete"
            if device == "cpu" and "skip" in statuses
            else "skip"
            if statuses == {"skip"}
            else "pass"
            if statuses == {"pass"}
            else "not_run"
        )
        report["device_results"][device] = {"status": status, "tests": outcomes}
    pipeline_module = sys.modules.get("test_amil_pipeline")
    report["pilot14"] = getattr(pipeline_module, "PILOT14_REPORT", None) or {
        "status": "not_run",
        "reason": "Real gate was not collected",
    }
    report["data_regressions"] = _regressions()
    required_skips = [
        item for item in result.outcomes if item["status"] == "skip" and not item["optional_device"]
    ]
    file_identity = report["pilot14_file"]
    file_stable = False
    if "sha256" in file_identity:
        try:
            after = Path(file_identity["path"]).stat()
            file_stable = (
                after.st_size == file_identity["size_bytes"]
                and after.st_mtime_ns == file_identity["mtime_ns"]
            )
        except OSError:
            pass
    report["pilot14_file"]["unchanged_during_validation"] = file_stable
    real = report["pilot14"]
    report["acceptance"] = {
        "amil_contract_and_integration": (
            result.wasSuccessful()
            and not required_skips
            and report["suite_coverage"]["status"] == "pass"
        ),
        "real_pilot14_14_wsi_42_full_forwards": (
            real.get("status") == "pass"
            and real.get("wsi_read") == 14
            and real.get("wsi_seen") == 14
            and real.get("views_compared") == 42
            and real.get("views_seen") == 42
            and real.get("forwards_succeeded") == 42
        ),
        "frozen_pilot14_and_official_encoder_provenance": (
            real.get("provenance_evidence", {}).get("status") == "pass"
            and real.get("provenance_evidence", {}).get("image_sources_hashed") == 14
        ),
        "pilot14_file_identified_and_stable": file_stable,
        "data_layer_regressions": report["data_regressions"]["status"] == "pass",
        "git_commit_identified": report["code"]["commit"] is not None,
    }
    report["status"] = "pass" if all(report["acceptance"].values()) else "fail"
    report["warnings"].extend(real.get("warnings", []))
    if report["code"].get("dirty"):
        report["warnings"].append(
            "Working tree was dirty; consult tested_file_sha256 "
            "and status_porcelain for the tested source identity."
        )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"AMIL acceptance: {report['status'].upper()}; report: {args.output.resolve()}")
    print(
        f"Real WSI read: {real.get('wsi_read', 0)}; full forwards: "
        f"{real.get('forwards_succeeded', 0)}/42; data regressions: "
        f"{report['data_regressions']['status']}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
