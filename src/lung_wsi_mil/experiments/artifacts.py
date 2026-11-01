"""Atomic experiment artifacts, strict JSON, and completed-stage verification."""

import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, rows, *, fieldnames=None):
    path, rows = Path(path), list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError("CSV fields must be supplied for an empty table.")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, run_fingerprint, seed, fold):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError(f"Unsupported checkpoint: {path}")
    if (state.get("run_fingerprint"), state.get("seed"), state.get("fold")) != (
        run_fingerprint,
        seed,
        fold,
    ):
        raise ValueError(f"Checkpoint identity differs from the experiment: {path}")
    return state


def seal_stage(directory, *, run_fingerprint, details):
    directory = Path(directory)
    files = {
        path.relative_to(directory).as_posix(): hash_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "complete.json" and ".tmp." not in path.name
    }
    atomic_json(
        directory / "complete.json",
        {
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "files": files,
            "details": details,
        },
    )


def verify_stage(directory, *, run_fingerprint):
    directory = Path(directory)
    manifest = read_json(directory / "complete.json")
    if manifest.get("run_fingerprint") != run_fingerprint or not manifest.get("files"):
        raise ValueError(f"Stale or empty completed stage: {directory}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "complete.json" and ".tmp." not in path.name
    }
    if actual != set(manifest["files"]):
        raise ValueError(f"Completed stage file coverage differs: {directory}")
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or not path.is_file()
            or hash_file(path) != expected
        ):
            raise ValueError(f"Completed stage artifact is missing or corrupt: {relative}")
    return manifest["details"]


@contextmanager
def run_lock(directory):
    path = Path(directory) / "run.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Experiment lock exists: {path}. Check the writer before removing a stale lock."
        ) from error
    try:
        with stream:
            json.dump({"pid": os.getpid()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)
