"""Preflight cohort identity, full feature contents, and all patient partitions."""

import hashlib
import io
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lung_wsi_mil.data.cross_validation import validate_fold_split
from lung_wsi_mil.data.splits import validate_patient_split
from lung_wsi_mil.experiments.artifacts import fingerprint, hash_file
from lung_wsi_mil.features.kimianet import OFFICIAL_SOURCE_COMMIT, OFFICIAL_WEIGHTS_SHA256

LABELS = {"Normal": 0, "Tumor": 1, "0": 0, "1": 1}
REFERENCE_INVENTORY_SHA256 = "33ef5ffa9710752ee94944511355c4a0bafa573788e2f48c045b978e32463916"


def load_metadata(path):
    raw = Path(path).read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    required = {"slide_id", "patient_id", "label"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty metadata needs columns {sorted(required)}: {path}")
    frame = frame[["slide_id", "patient_id", "label"]].copy()
    for name in ("slide_id", "patient_id"):
        if any(
            not value or value.strip() != value or value.upper() in {"NA", "NULL", "NONE"}
            for value in frame[name]
        ):
            raise ValueError(f"Invalid {name} values: {path}")
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}
        for value in frame["slide_id"]
    ):
        raise ValueError(f"Slide IDs must be safe single path components: {path}")
    if frame.slide_id.duplicated().any():
        raise ValueError(f"Duplicate slide IDs: {path}")
    if not frame.label.isin(LABELS).all():
        raise ValueError(f"Unknown binary label: {path}")
    frame["label"] = frame.label.map(LABELS).astype(int)
    frame = frame.sort_values("slide_id").reset_index(drop=True)
    frame.attrs["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return frame


def _match_inventory(partition, inventory, path):
    by_id = inventory.set_index("slide_id")
    for row in partition.itertuples():
        if row.slide_id not in by_id.index:
            raise ValueError(f"Unexpected split slide {row.slide_id}: {path}")
        expected = by_id.loc[row.slide_id]
        if row.patient_id != expected.patient_id or row.label != expected.label:
            raise ValueError(f"Split patient/label conflicts with inventory: {row.slide_id}")
    if set(partition.label) != {0, 1}:
        raise ValueError(f"Both classes are required for meaningful validation/test AUROC: {path}")


def validate_partitions(inventory, split_root, config):
    """Every patient is isolated; each Train/CV slide is validated exactly once."""
    split_root, partitions, hashes = Path(split_root), {}, {}
    for seed in config.seeds:
        folder = split_root / f"seed_{seed}"
        frames = {}
        for name in [
            "train_cv",
            "test",
            *[f"fold_{fold}_{role}" for fold in range(1, 6) for role in ("train", "val")],
        ]:
            path = folder / f"{name}.csv"
            frame = load_metadata(path)
            if hash_file(path) != frame.attrs["source_sha256"]:
                raise ValueError(f"Split CSV changed during preflight: {path}")
            _match_inventory(frame, inventory, path)
            frames[name] = frame
            hashes[f"splits/seed_{seed}/{name}.csv"] = frame.attrs["source_sha256"]
        validate_patient_split(inventory, frames["train_cv"], frames["test"])
        validations = []
        for fold in range(1, 6):
            train, val = frames[f"fold_{fold}_train"], frames[f"fold_{fold}_val"]
            validate_fold_split(frames["train_cv"], train, val)
            if set(train.patient_id) & set(frames["test"].patient_id) or set(val.patient_id) & set(
                frames["test"].patient_id
            ):
                raise ValueError("Outer-test patient leakage into an inner fold.")
            validations.extend(val.slide_id)
        if Counter(validations) != Counter({slide: 1 for slide in frames["train_cv"].slide_id}):
            raise ValueError("Inner validation folds must partition Train/CV exactly once.")
        partitions[seed] = frames
    return partitions, hashes


def _digest_embeddings(dataset):
    digest = hashlib.sha256()
    for view in range(3):
        for start in range(0, dataset.shape[1], 256):
            values = dataset[view, start : start + 256]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite embeddings: {dataset.name}")
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def validate_features(path, inventory, *, inventory_sha256, allow_unverified=False):
    """Read every embedding in chunks; official provenance is required by default.

    Source image bytes are not needed for training. Their recorded hashes remain
    attached to the immutable feature cache, but are not re-certified here.
    """
    import json

    path = Path(path).expanduser().resolve()
    before = path.stat()
    records = {row.slide_id: row for row in inventory.itertuples()}
    verified, total_patches, sources = False, 0, {}
    with h5py.File(path, "r") as file:
        if set(file) != set(records):
            raise ValueError(
                "Feature cache must cover exactly the inventory; extract the full cohort first."
            )
        if not allow_unverified:
            try:
                provenance = json.loads(file.attrs["provenance_json"])
                if (
                    file.attrs.get("status") != "complete"
                    or file.attrs.get("format_version") != "1.0"
                    or fingerprint(provenance) != file.attrs.get("fingerprint")
                    or fingerprint(provenance["config"]) != file.attrs.get("config_sha256")
                    or fingerprint(provenance["code"]) != file.attrs.get("code_sha256")
                    or provenance.get("weights_sha256") != OFFICIAL_WEIGHTS_SHA256
                    or file.attrs.get("weights_sha256") != OFFICIAL_WEIGHTS_SHA256
                    or provenance.get("encoder_source_commit") != OFFICIAL_SOURCE_COMMIT
                    or provenance.get("inventory_sha256") != inventory_sha256
                ):
                    raise ValueError(
                        "Official feature provenance/config/inventory checksum mismatch."
                    )
                source_rows = provenance["sources"]
                sources = {source["slide_id"]: source for source in source_rows}
                if len(sources) != len(source_rows) or set(sources) != set(records):
                    raise ValueError("Feature provenance source coverage differs from inventory.")
                verified = True
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Official feature provenance is required; unverified inputs need an explicit override."
                ) from error
        for slide, row in records.items():
            group = file[slide]
            if not isinstance(group, h5py.Group) or not {
                "embeddings",
                "coords",
                "label",
                "case_id",
            }.issubset(group):
                raise ValueError(f"Incomplete feature slide group: {slide}")
            embeddings, coords = group["embeddings"], group["coords"]
            if (
                embeddings.ndim != 3
                or embeddings.shape[0] != 3
                or embeddings.shape[2] != 1024
                or embeddings.shape[1] == 0
                or embeddings.dtype != np.dtype("float32")
            ):
                raise ValueError(f"{slide}: expected nonempty float32 [3,N,1024] embeddings.")
            n = embeddings.shape[1]
            if coords.shape != (n, 2) or coords.dtype != np.dtype("int64") or (coords[:] < 0).any():
                raise ValueError(f"{slide}: expected aligned nonnegative int64 [N,2] coordinates.")
            if (
                group["label"].shape != ()
                or group["label"][()] != row.label
                or group["case_id"].shape != ()
                or group["case_id"].asstr()[()] != row.patient_id
            ):
                raise ValueError(f"{slide}: feature label/patient conflicts with inventory.")
            embedding_digest = _digest_embeddings(embeddings)
            if verified:
                source = sources[slide]
                if (
                    source.get("case_id") != row.patient_id
                    or source.get("label") != row.label
                    or source.get("source_label") != ("Tumor" if row.label else "Normal")
                    or source.get("n") != n
                    or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
                    or group.attrs.get("source_sha256") != source.get("sha256")
                    or group.attrs.get("source_run_fingerprint") != source.get("run_fingerprint")
                    or group.attrs.get("embeddings_sha256") != embedding_digest
                    or group.attrs.get("coords_sha256")
                    != hashlib.sha256(coords[:].tobytes()).hexdigest()
                ):
                    raise ValueError(f"{slide}: feature content/source checksum mismatch.")
            total_patches += n
    digest = hash_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Feature cache changed during preflight.")
    return {
        "sha256": digest,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "slides": len(records),
        "patients": int(inventory.patient_id.nunique()),
        "patches": total_patches,
        "official_provenance_verified": verified,
        "input_scope": "provenance-verified features" if verified else "SYNTHETIC_OR_UNVERIFIED",
        "raw_image_sources_reverified": False,
    }
