"""Streaming, provenance-checked KimiaNet cache from saved augmentation HDF5.

Image tensors are bounded by one three-view chunk.  The combined cache is
published only after every selected slide and embedding has been validated.
"""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

import h5py
import numpy as np
import torch

from src.augmentation.storage import load_augmented_chunk, validate_augmented_hdf5
from lung_wsi_mil.features.kimianet import (
    KimiaNetFeatureExtractor,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_WEIGHTS_SHA256,
    OFFICIAL_WEIGHTS_URL,
    load_kimianet_weights,
)


FORMAT_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "slide_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV requires slide_id column: {path}")
        rows = list(reader)
    ids = [row["slide_id"].strip() for row in rows]
    if not ids or any(not value or "/" in value or "\\" in value or value in {".", ".."}
                      for value in ids):
        raise ValueError(f"CSV has empty or invalid slide IDs: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"CSV has duplicate slide IDs: {path}")
    for row, slide_id in zip(rows, ids):
        row["slide_id"] = slide_id
    return rows


def selected_records(inventory, selection=None):
    """Use frozen inventory identity/labels and reject conflicting selection metadata."""
    inventory_rows = _csv_rows(inventory)
    indexed = {row["slide_id"]: row for row in inventory_rows}
    selected = _csv_rows(selection) if selection is not None else inventory_rows
    records = []
    for selected_row in selected:
        slide_id = selected_row["slide_id"]
        if slide_id not in indexed:
            raise ValueError(f"Selected slide is missing from inventory: {slide_id}")
        row = indexed[slide_id]
        label = row.get("label", "").strip()
        patient = row.get("patient_id", "").strip()
        if label not in {"Normal", "Tumor"} or not patient:
            raise ValueError(f"{slide_id}: inventory needs Normal/Tumor label and patient_id")
        for key, expected in (("label", label), ("patient_id", patient)):
            if selected_row.get(key, "").strip() not in {"", expected}:
                raise ValueError(f"{slide_id}: selection {key} conflicts with inventory")
        records.append({"slide_id": slide_id, "case_id": patient,
                        "source_label": label, "label": int(label == "Tumor")})
    return records


def _code_identity():
    paths = ("src/features/hdf5_pipeline.py", "src/lung_wsi_mil/features/kimianet.py",
             "src/augmentation/storage.py", "src/augmentation/data_augmentations.py")
    return {name: hash_file(PROJECT_ROOT / name) for name in paths}


def _load_encoder(weights, device):
    model = load_kimianet_weights(
        KimiaNetFeatureExtractor(), weights,
        expected_sha256=OFFICIAL_WEIGHTS_SHA256,
    )
    model.requires_grad_(False)
    return model.to(device).eval()


def _source_records(patch_root, records):
    sources = []
    for record in records:
        slide_id = record["slide_id"]
        path = Path(patch_root) / "slides" / slide_id / "augmentation" / "patch_views.h5"
        metadata = validate_augmented_hdf5(path, expected={
            "slide_id": slide_id, "patient_id": record["case_id"],
            "label": record["source_label"]}, verify_values=True)
        if metadata["patch_count"] <= 0:
            raise ValueError(f"{slide_id}: empty bag; no selected slide may be silently excluded")
        before = path.stat()
        digest = hash_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"{slide_id}: source changed during hashing")
        sources.append({**record, "path": str(path.resolve()),
                        "sha256": digest, "size_bytes": after.st_size,
                        "mtime_ns": after.st_mtime_ns, "n": metadata["patch_count"],
                        "run_fingerprint": metadata["run_fingerprint"]})
    return sources


def _embedding_digest(dataset, batch_size=256):
    digest = hashlib.sha256()
    for view in range(3):
        for start in range(0, dataset.shape[1], batch_size):
            block = dataset[view, start:start + batch_size]
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite embeddings in {dataset.name}")
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest()


def validate_feature_cache(path, *, fingerprint, sources):
    """Validate complete content, provenance, and hashes before publication/resume."""
    with h5py.File(path, "r") as file:
        if file.attrs.get("format_version") != FORMAT_VERSION or file.attrs.get("status") != "complete":
            raise ValueError("Feature cache is incomplete or has an unsupported format")
        if file.attrs.get("fingerprint") != fingerprint:
            raise ValueError("Stale feature cache: source/config/code/weights fingerprint differs")
        try:
            provenance = json.loads(file.attrs["provenance_json"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Feature cache provenance is missing or invalid") from error
        if not isinstance(provenance, dict):
            raise ValueError("Feature cache provenance must be a JSON object")
        if (_fingerprint(provenance) != fingerprint or
                file.attrs.get("config_sha256") != _fingerprint(provenance.get("config")) or
                file.attrs.get("code_sha256") != _fingerprint(provenance.get("code")) or
                file.attrs.get("weights_sha256") != provenance.get("weights_sha256")):
            raise ValueError("Feature cache provenance checksum mismatch")
        if set(file) != {source["slide_id"] for source in sources}:
            raise ValueError("Feature cache slide set differs from selection")
        for source in sources:
            slide_id = source["slide_id"]
            group = file[slide_id]
            if set(group) != {"embeddings", "coords", "label", "case_id"}:
                raise ValueError(f"{slide_id}: incomplete or unexpected cache datasets")
            n = source["n"]
            features = group["embeddings"]
            if features.shape != (3, n, 1024) or features.dtype != np.dtype("float32"):
                raise ValueError(f"{slide_id}: invalid feature shape or dtype")
            coords = group["coords"]
            if coords.shape != (n, 2) or coords.dtype != np.dtype("int64"):
                raise ValueError(f"{slide_id}: invalid coordinate shape or dtype")
            coordinates = coords[:]
            with h5py.File(source["path"], "r") as original:
                if not np.array_equal(coordinates, original["coords"][:]):
                    raise ValueError(f"{slide_id}: coordinates differ from source")
            if (coordinates < 0).any():
                raise ValueError(f"{slide_id}: negative real coordinates")
            if (group["label"].shape != () or group["label"][()] != source["label"] or
                    group["case_id"].shape != () or group["case_id"].asstr()[()] != source["case_id"]):
                raise ValueError(f"{slide_id}: label or case_id differs from inventory")
            if (group.attrs.get("source_sha256") != source["sha256"] or
                    group.attrs.get("source_run_fingerprint") != source["run_fingerprint"]):
                raise ValueError(f"{slide_id}: source provenance mismatch")
            if _embedding_digest(features) != group.attrs.get("embeddings_sha256"):
                raise ValueError(f"{slide_id}: embedding checksum mismatch")
            if hashlib.sha256(coordinates.tobytes()).hexdigest() != group.attrs.get("coords_sha256"):
                raise ValueError(f"{slide_id}: coordinate checksum mismatch")
    return {"slides": len(sources), "patches": sum(source["n"] for source in sources),
            "views": 3 * len(sources), "patch_views": 3 * sum(source["n"] for source in sources)}


def extract_feature_cache(*, patch_root, inventory, weights, output, selection=None,
                          batch_size=8, device="cpu", resume=False):
    """Create one genuine, frozen KimiaNet cache; reuse only an identical valid cache."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output = Path(output).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"KimiaNet weights not found: {weights}")
    if output.exists() and not resume:
        raise FileExistsError(f"Output already exists; use --resume for verified reuse: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise FileExistsError(f"Output is locked by another/interrupted extraction: {lock_path}") from error
    stage = output.with_name(output.name + f".partial.{uuid.uuid4().hex}")
    try:
        with lock:
            lock.write(_json({"pid": os.getpid(), "output": str(output)}))
        code = _code_identity()
        inventory_hash = hash_file(inventory)
        selection_hash = hash_file(selection) if selection is not None else None
        print("Validating and hashing selected augmentation sources...", flush=True)
        sources = _source_records(patch_root, selected_records(inventory, selection))
        if code != _code_identity():
            raise ValueError("Feature pipeline code changed during source validation")
        if (hash_file(inventory) != inventory_hash or
                (selection is not None and hash_file(selection) != selection_hash)):
            raise ValueError("Inventory or selection changed during source validation")
        config = {"format_version": FORMAT_VERSION, "batch_size": batch_size,
                  "device": device, "feature_dim": 1024, "patch_size": 512,
                  "representations": ["Original", "Aug1", "Aug2"],
                  "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
                  "normalization": "src.augmentation.storage.load_augmented_chunk",
                  "matmul_precision": torch.get_float32_matmul_precision(),
                  "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                  "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                  "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                  "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        identity = {"config": config, "code": code, "weights_sha256": hash_file(weights),
                    "encoder_source_commit": OFFICIAL_SOURCE_COMMIT,
                    "encoder_weights_url": OFFICIAL_WEIGHTS_URL,
                    "inventory_sha256": inventory_hash,
                    "selection_sha256": selection_hash,
                    "sources": [{key: value for key, value in source.items()
                                 if key not in {"path", "mtime_ns"}} for source in sources]}
        fingerprint = _fingerprint(identity)
        if output.exists():
            summary = validate_feature_cache(output, fingerprint=fingerprint, sources=sources)
            return {"status": "resumed", "output": str(output), "fingerprint": fingerprint,
                    "sha256": hash_file(output), **summary}
        print(f"Loading genuine KimiaNet weights: {weights}", flush=True)
        model = _load_encoder(weights, torch.device(device))
        if getattr(model, "feature_dim", None) != 1024:
            raise ValueError("KimiaNet must expose feature_dim=1024")
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("KimiaNet encoder must be frozen and in evaluation mode")
        with h5py.File(stage, "w") as target:
            target.attrs.update(format_version=FORMAT_VERSION, status="writing",
                                fingerprint=fingerprint, provenance_json=_json(identity),
                                config_sha256=_fingerprint(config), code_sha256=_fingerprint(code),
                                weights_sha256=identity["weights_sha256"], weights_path=str(weights),
                                created_utc=datetime.now(timezone.utc).isoformat(),
                                python=platform.python_version(), platform=platform.platform(),
                                device_name=(torch.cuda.get_device_name() if device == "cuda" else "cpu"),
                                embedding_layout="representation,patch,feature",
                                feature_dim=1024)
            for index, source in enumerate(sources, 1):
                slide_id, n = source["slide_id"], source["n"]
                print(f"[{index}/{len(sources)}] {slide_id}: {n} patches, 3 views", flush=True)
                group = target.create_group(slide_id)
                group.attrs.update(source_sha256=source["sha256"], source_path=source["path"],
                                   source_run_fingerprint=source["run_fingerprint"])
                features = group.create_dataset("embeddings", shape=(3, n, 1024),
                                                dtype=np.float32, chunks=(1, min(batch_size, n), 1024),
                                                compression="gzip", compression_opts=1)
                coordinates = group.create_dataset("coords", shape=(n, 2), dtype=np.int64)
                group.create_dataset("label", data=np.int64(source["label"]))
                group.create_dataset("case_id", data=source["case_id"].encode("utf-8"))
                with torch.inference_mode():
                    for start in range(0, n, batch_size):
                        stop = min(n, start + batch_size)
                        transformed, coords, _ = load_augmented_chunk(source["path"], start, stop)
                        coordinates[start:stop] = coords
                        for view in range(3):
                            embeddings, _ = model(transformed[view].to(device))
                            if embeddings.shape != (stop - start, 1024):
                                raise ValueError(f"{slide_id}: invalid KimiaNet output shape")
                            if embeddings.dtype != torch.float32 or not bool(torch.isfinite(embeddings).all()):
                                raise ValueError(f"{slide_id}: expected finite float32 KimiaNet embeddings")
                            features[view, start:stop] = embeddings.cpu().numpy()
                        del transformed, embeddings
                group.attrs["embeddings_sha256"] = _embedding_digest(features)
                group.attrs["coords_sha256"] = hashlib.sha256(coordinates[:].tobytes()).hexdigest()
                stat = Path(source["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                    raise ValueError(f"{slide_id}: source changed during extraction")
                target.flush()
            if hash_file(weights) != identity["weights_sha256"]:
                raise ValueError("KimiaNet weights changed during extraction")
            if code != _code_identity():
                raise ValueError("Feature pipeline code changed during extraction")
            if (hash_file(inventory) != inventory_hash or
                    (selection is not None and hash_file(selection) != selection_hash)):
                raise ValueError("Inventory or selection changed during extraction")
            target.attrs["status"] = "complete"
        summary = validate_feature_cache(stage, fingerprint=fingerprint, sources=sources)
        os.replace(stage, output)
        return {"status": "created", "output": str(output), "fingerprint": fingerprint,
                "sha256": hash_file(output), **summary}
    finally:
        if stage.exists():
            stage.unlink()
        lock_path.unlink(missing_ok=True)
