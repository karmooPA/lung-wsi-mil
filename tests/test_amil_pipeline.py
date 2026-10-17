"""Frozen HDF5 -> Dataset -> DataLoader -> AMIL contract.

The padding helper deliberately lives in this test module.  It is not a
training loop or a production padding API.  The real acceptance test requires
PILOT14_H5; skipping it during local development does not complete the gate.
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from src.data.mil_dataloader import create_mil_dataloader
from src.data.mil_hdf5_dataset import MILHDF5Dataset
from src.mil.amil import AMIL


PILOT14_REPORT = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def trim_test_padding(padded_bag, padded_coords):
    """Remove only aligned, contiguous, complete -Inf/-1 tail sentinels.

    Rejecting corrupted nonfinite rows first prevents a finite-row filter from
    silently dropping real, damaged embeddings and reporting a false success.
    """
    _require(padded_bag.ndim == 2 and padded_bag.shape[1] == 1024,
             "Expected one padded [Nmax, 1024] bag")
    _require(padded_coords.shape == (padded_bag.shape[0], 2),
             "Padded coordinate shape must match bag rows")
    valid = torch.isfinite(padded_bag).all(dim=1)
    padding = torch.isneginf(padded_bag).all(dim=1)
    _require(bool((valid | padding).all()),
             "Rows must be fully finite or complete -Inf padding")
    n = int(valid.sum().item())
    _require(n > 0, "A trimmed bag must contain at least one real instance")
    _require(bool(valid[:n].all()) and not bool(valid[n:].any()),
             "Padding must be a contiguous tail")
    coordinate_padding = (padded_coords == -1).all(dim=1)
    _require(torch.equal(coordinate_padding, padding),
             "Embedding and coordinate padding sentinels must align")
    _require(bool((padded_coords[:n] >= 0).all()),
             "Real coordinates must be nonnegative")
    clean = padded_bag[valid]
    _require(clean.shape == (n, 1024) and bool(torch.isfinite(clean).all()),
             "Trimmed bag must be nonempty, finite, and [N, 1024]")
    return clean


def _check_output(model, bag, context):
    output = model(bag)
    _require(isinstance(output, tuple) and len(output) == 3,
             f"{context}: forward must return exactly (Y_prob, Y_hat, A)")
    y_prob, y_hat, attention = output
    _require(y_prob.shape == (1, 1), f"{context}: incorrect Y_prob shape")
    _require(y_hat.shape == (1, 1), f"{context}: incorrect Y_hat shape")
    _require(attention.shape == (1, bag.shape[0]),
             f"{context}: incorrect attention shape")
    for name, tensor in zip(("Y_prob", "Y_hat", "A"), output):
        _require(bool(torch.isfinite(tensor).all()),
                 f"{context}: nonfinite {name}")
    _require(bool(((y_prob >= 0) & (y_prob <= 1)).all()),
             f"{context}: probability outside [0, 1]")
    _require(bool(((y_hat == 0) | (y_hat == 1)).all()),
             f"{context}: prediction is not binary")
    _require(torch.equal(y_hat, (y_prob >= model.threshold).to(y_hat.dtype)),
             f"{context}: prediction threshold mismatch")
    _require(bool((attention >= 0).all()), f"{context}: negative attention")
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(1),
                               atol=1e-5, rtol=1e-5,
                               msg=f"{context}: attention must sum to one")


def validate_real_pilot_provenance(path):
    """Bind the real gate to the frozen cohort and complete KimiaNet cache.

    Generic HDF5 integration fixtures deliberately bypass this check. A real
    acceptance run must additionally retain readable, matching image sources
    and provenance from the pinned official encoder extraction pipeline.
    """
    from src.augmentation.storage import validate_augmented_hdf5
    from src.features.hdf5_pipeline import (
        _code_identity, hash_file, selected_records, validate_feature_cache,
    )
    from src.features.kimianet import (
        OFFICIAL_SOURCE_COMMIT, OFFICIAL_WEIGHTS_SHA256, OFFICIAL_WEIGHTS_URL,
    )

    inventory = PROJECT_ROOT / "metadata/dataset_inventory.csv"
    selection = PROJECT_ROOT / "metadata/pilot14_selection.csv"
    expected = selected_records(inventory, selection)
    expected_by_id = {record["slide_id"]: record for record in expected}
    _require(len(expected) == 14 and len({r["case_id"] for r in expected}) == 14,
             "Frozen Pilot14 must contain exactly 14 distinct patients")
    _require(Counter(record["label"] for record in expected) == {0: 7, 1: 7},
             "Frozen Pilot14 must contain 7 Normal and 7 Tumor slides")
    sources = []
    with h5py.File(path, "r") as cache:
        _require(set(cache) == set(expected_by_id),
                 "Real gate slide IDs must equal metadata/pilot14_selection.csv")
        _require(cache.attrs.get("status") == "complete",
                 "Real gate requires a complete, provenance-checked feature cache")
        try:
            provenance = json.loads(cache.attrs["provenance_json"])
        except (KeyError, TypeError, ValueError) as error:
            raise AssertionError("Real gate requires feature extraction provenance") from error
        _require(isinstance(provenance, dict), "Feature provenance must be an object")
        _require(provenance.get("weights_sha256") == OFFICIAL_WEIGHTS_SHA256 and
                 provenance.get("encoder_source_commit") == OFFICIAL_SOURCE_COMMIT and
                 provenance.get("encoder_weights_url") == OFFICIAL_WEIGHTS_URL,
                 "Real gate requires the pinned official KimiaNet weights/source")
        _require(provenance.get("code") == _code_identity(),
                 "Feature extraction code differs from recorded generating code")
        _require(provenance.get("inventory_sha256") == hash_file(inventory) and
                 provenance.get("selection_sha256") == hash_file(selection),
                 "Feature inventory/selection hashes differ from frozen Pilot14 metadata")
        recorded_sources = provenance.get("sources")
        _require(isinstance(recorded_sources, list) and len(recorded_sources) == 14 and
                 all(isinstance(source, dict) for source in recorded_sources),
                 "Feature provenance must identify all 14 image sources")
        _require(Counter(source.get("slide_id") for source in recorded_sources) ==
                 Counter({slide_id: 1 for slide_id in expected_by_id}),
                 "Feature provenance image sources differ from frozen Pilot14")
        for source in recorded_sources:
            slide_id = source["slide_id"]
            expected_record = expected_by_id[slide_id]
            _require(all(source.get(key) == value for key, value in expected_record.items()),
                     f"{slide_id}: feature source identity/label differs from inventory")
            group = cache[slide_id]
            _require(group["case_id"].shape == () and
                     group["case_id"].asstr()[()] == expected_record["case_id"] and
                     group["label"].shape == () and
                     group["label"][()] == expected_record["label"],
                     f"{slide_id}: feature case_id/label differs from inventory")
            source_path = Path(str(group.attrs.get("source_path", ""))).expanduser()
            _require(source_path.is_absolute() and source_path.is_file(),
                     f"{slide_id}: recorded augmentation image source is unavailable")
            before = source_path.stat()
            metadata = validate_augmented_hdf5(source_path, expected={
                "slide_id": slide_id, "patient_id": expected_record["case_id"],
                "label": expected_record["source_label"]}, verify_values=True)
            _require(metadata["patch_count"] > 0 and
                     metadata["patch_count"] == source.get("n") and
                     metadata["run_fingerprint"] == source.get("run_fingerprint"),
                     f"{slide_id}: source patch count or generating fingerprint differs")
            _require(before.st_size == source.get("size_bytes") and
                     hash_file(source_path) == source.get("sha256"),
                     f"{slide_id}: augmentation source size/SHA256 differs from provenance")
            after = source_path.stat()
            _require((before.st_size, before.st_mtime_ns) ==
                     (after.st_size, after.st_mtime_ns),
                     f"{slide_id}: augmentation source changed during validation")
            sources.append({**source, "path": str(source_path.resolve())})
        fingerprint = cache.attrs.get("fingerprint")
    # Recomputes provenance/config/code hashes, embedding checksums, coordinates,
    # source links, exact dataset schema, and per-slide patient/label identity.
    summary = validate_feature_cache(path, fingerprint=fingerprint, sources=sources)
    return {"status": "pass", "frozen_selection_sha256": hash_file(selection),
            "inventory_sha256": hash_file(inventory),
            "weights_sha256": OFFICIAL_WEIGHTS_SHA256,
            "encoder_source_commit": OFFICIAL_SOURCE_COMMIT,
            "feature_fingerprint": fingerprint,
            "image_sources_hashed": len(sources), **summary}


def validate_hdf5_pipeline(path, *, expected_slides=14,
                           expected_label_counts=None, report=None):
    """Read every source value and forward every view on CPU, without sampling.

    Only small metadata records survive each pass; embeddings are held for one
    source slide or one DataLoader batch at a time.  A caller-supplied report is
    updated as checks complete so failed CLI runs retain honest partial counts.
    """
    report = {} if report is None else report
    report.update(status="running", path=str(Path(path).expanduser().resolve()),
                  wsi_read=0, views_compared=0, forwards_succeeded=0,
                  batch_size=4, shuffle=False, device="cpu", warnings=[])
    dataset = MILHDF5Dataset(path)
    _require(len(dataset) == expected_slides,
             f"Expected {expected_slides} WSI, found {len(dataset)}")
    records = {}
    label_counts = Counter()
    with h5py.File(path, "r") as source:
        for index in range(len(dataset)):
            embeddings, coords, label, slide_id, case_id = dataset[index]
            context = f"source slide {slide_id}"
            _require(isinstance(slide_id, str) and bool(slide_id.strip()),
                     f"{context}: missing slide_id")
            _require(isinstance(case_id, str) and bool(case_id.strip()),
                     f"{context}: missing case_id")
            _require(slide_id not in records, f"{context}: repeated slide_id")
            n = embeddings.shape[1]
            _require(embeddings.shape == (3, n, 1024) and n > 0,
                     f"{context}: expected nonempty [3,N,1024] embeddings")
            _require(embeddings.dtype == torch.float32,
                     f"{context}: Dataset must produce float32 embeddings")
            _require(bool(torch.isfinite(embeddings).all()),
                     f"{context}: nonfinite embeddings")
            _require(coords.shape == (n, 2) and coords.dtype == torch.int64,
                     f"{context}: expected int64 [N,2] coordinates")
            _require(bool((coords >= 0).all()),
                     f"{context}: negative real coordinates")
            _require(label.ndim == 0 and label.item() in (0, 1),
                     f"{context}: label must be a binary scalar")
            group = source[slide_id]
            _require(group["embeddings"].dtype.kind == "f",
                     f"{context}: stored embeddings must be floating point")
            _require(group["coords"].dtype.kind in "iu",
                     f"{context}: stored coordinates must be integral")
            raw_embeddings = group["embeddings"][:]
            _require(bool(np.isfinite(raw_embeddings).all()),
                     f"{context}: nonfinite values in stored embeddings")
            raw_coords = group["coords"][:]
            _require(bool((raw_coords >= 0).all()),
                     f"{context}: negative stored coordinates")
            _require(torch.equal(embeddings, torch.tensor(raw_embeddings,
                                                        dtype=torch.float32)),
                     f"{context}: Dataset embeddings differ from source")
            _require(torch.equal(coords, torch.tensor(raw_coords,
                                                    dtype=torch.int64)),
                     f"{context}: Dataset coordinates differ from source")
            if group["embeddings"].dtype != np.dtype("float32"):
                report["warnings"].append(
                    f"{slide_id}: stored {group['embeddings'].dtype} embeddings "
                    "are converted to float32 by the frozen Dataset")
            records[slide_id] = {"n": n, "label": int(label.item()),
                                 "case_id": case_id}
            label_counts[int(label.item())] += 1
            report["wsi_read"] += 1
            del embeddings, coords, raw_embeddings, raw_coords

        report["label_counts"] = dict(sorted(label_counts.items()))
        report["min_n"] = min(record["n"] for record in records.values())
        report["max_n"] = max(record["n"] for record in records.values())
        report["slides"] = [{"slide_id": slide_id, **record}
                            for slide_id, record in records.items()]
        if expected_label_counts is not None:
            _require(dict(label_counts) == expected_label_counts,
                     "Pilot label balance mismatch: expected "
                     f"{expected_label_counts}, observed {dict(label_counts)}. "
                     "Reconcile the cohort with its approved metadata.")

        loader = create_mil_dataloader(dataset, batch_size=4, shuffle=False,
                                       num_workers=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            model = AMIL().cpu().eval()
        seen = Counter()
        seen_order = []
        batch_shapes = []
        padded_rows = 0
        with torch.no_grad():
            for batch_number, batch in enumerate(loader):
                bags, coordinates, labels, slide_ids, case_ids, augmented = batch
                expected_ids = dataset.slide_ids[batch_number * 4:
                                                  batch_number * 4 + 4]
                expanded_ids = [slide_id for slide_id in expected_ids
                                for _ in range(3)]
                nmax = max(records[slide_id]["n"] for slide_id in expected_ids)
                count = 3 * len(expected_ids)
                _require(bags.shape == (count, nmax, 1024),
                         f"batch {batch_number}: incorrect 3B/Nmax/feature shape")
                _require(coordinates.shape == (count, nmax, 2),
                         f"batch {batch_number}: incorrect coordinate shape")
                _require(labels.shape == (count,), "Incorrect label batch shape")
                _require(slide_ids == expanded_ids,
                         f"batch {batch_number}: missing, reordered, or repeated WSI")
                _require(len(case_ids) == count, "Incorrect case_id count")
                _require(augmented.dtype == torch.bool and
                         augmented.tolist() == [False, True, True] * len(expected_ids),
                         f"batch {batch_number}: Original/Aug1/Aug2 flags changed")
                batch_shapes.append(list(bags.shape))
                for row, slide_id in enumerate(slide_ids):
                    view = row % 3
                    record = records[slide_id]
                    n = record["n"]
                    context = f"{slide_id}/view{view}"
                    _require(case_ids[row] == record["case_id"],
                             f"{context}: case_id changed")
                    _require(labels[row].item() == record["label"],
                             f"{context}: label changed")
                    _require(bool(torch.isfinite(bags[row, :n]).all()),
                             f"{context}: damaged real rows")
                    _require(bool(torch.isneginf(bags[row, n:]).all()),
                             f"{context}: embedding tail must be entirely -Inf")
                    _require(bool((coordinates[row, n:] == -1).all()),
                             f"{context}: coordinate tail must be exactly [-1,-1]")
                    clean = trim_test_padding(bags[row], coordinates[row])
                    _require(clean.shape == (n, 1024),
                             f"{context}: trim lost or added real rows")
                    expected_bag = torch.tensor(
                        source[slide_id]["embeddings"][view], dtype=torch.float32)
                    expected_coords = torch.tensor(
                        source[slide_id]["coords"][:], dtype=torch.int64)
                    _require(torch.equal(clean, expected_bag),
                             f"{context}: collate/trim altered source embeddings")
                    _require(torch.equal(coordinates[row, :n], expected_coords),
                             f"{context}: collate altered source coordinates")
                    if view:
                        _require(torch.equal(coordinates[row], coordinates[row - view]),
                                 f"{context}: coordinates differ across views")
                    report["views_compared"] += 1
                    _check_output(model, clean, context)
                    report["forwards_succeeded"] += 1
                    seen[slide_id] += 1
                    seen_order.append((slide_id, view))
                    padded_rows += nmax - n
                    del clean, expected_bag, expected_coords
    _require(seen == Counter({slide_id: 3 for slide_id in records}),
             "Every WSI must occur exactly three times")
    _require(len(seen_order) == 3 * expected_slides,
             "Not all Original/Aug1/Aug2 views were processed")
    report.update(status="pass", wsi_seen=len(seen),
                  views_seen=len(seen_order), batch_shapes=batch_shapes,
                  padding_rows_checked=padded_rows,
                  source_equality="exact_after_frozen_dataset_dtype_conversion",
                  all_inputs_outputs_finite=True,
                  all_attention_normalized=True)
    return report


def _write_fixture(path, sizes=(1, 3, 7, 2)):
    generator = np.random.default_rng(2026)
    with h5py.File(path, "w") as target:
        for index, n in enumerate(sizes):
            group = target.create_group(f"slide_{index:02d}")
            group.create_dataset("embeddings", data=generator.normal(
                size=(3, n, 1024)).astype(np.float32))
            group.create_dataset("coords", data=np.column_stack((
                np.arange(n) * 512, np.arange(n) * 256)).astype(np.int64))
            group.create_dataset("label", data=np.int64(index % 2))
            group.create_dataset("case_id", data=f"case_{index:02d}".encode())


class AMILPipelineTests(unittest.TestCase):
    def test_real_gate_rejects_balanced_synthetic_hdf5(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic14.h5"
            _write_fixture(path, sizes=tuple(range(1, 15)))
            with self.assertRaisesRegex(AssertionError, "slide IDs"):
                validate_real_pilot_provenance(path)
            from src.features.hdf5_pipeline import selected_records
            expected = selected_records(PROJECT_ROOT / "metadata/dataset_inventory.csv",
                                        PROJECT_ROOT / "metadata/pilot14_selection.csv")
            with h5py.File(path, "r+") as cache:
                for name, record in zip(sorted(cache), expected):
                    cache.move(name, record["slide_id"])
                cache.attrs["status"] = "complete"
            with self.assertRaisesRegex(AssertionError, "provenance"):
                validate_real_pilot_provenance(path)

    def test_small_hdf5_all_views_match_source_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.h5"
            _write_fixture(path)
            report = validate_hdf5_pipeline(path, expected_slides=4,
                                            expected_label_counts={0: 2, 1: 2})
        self.assertEqual(report["wsi_read"], 4)
        self.assertEqual(report["views_seen"], 12)
        self.assertEqual(report["views_compared"], 12)
        self.assertEqual(report["forwards_succeeded"], 12)
        self.assertEqual(report["batch_shapes"], [[12, 7, 1024]])
        self.assertGreater(report["padding_rows_checked"], 0)
        self.assertEqual((report["min_n"], report["max_n"]), (1, 7))

    def test_equal_size_bags_need_no_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "equal.h5"
            _write_fixture(path, sizes=(3, 3))
            report = validate_hdf5_pipeline(path, expected_slides=2,
                                            expected_label_counts={0: 1, 1: 1})
        self.assertEqual(report["padding_rows_checked"], 0)
        self.assertEqual(report["forwards_succeeded"], 6)

    def test_partial_last_batch_keeps_all_views(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial_batch.h5"
            _write_fixture(path, sizes=(1, 3, 7, 2, 9, 4))
            report = validate_hdf5_pipeline(path, expected_slides=6,
                                            expected_label_counts={0: 3, 1: 3})
        self.assertEqual(report["batch_shapes"], [[12, 7, 1024], [6, 9, 1024]])
        self.assertEqual(report["forwards_succeeded"], 18)

    def test_gate_detects_changed_source_views_and_metadata(self):
        original_loader = create_mil_dataloader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unaltered_source.h5"
            _write_fixture(path)
            for corruption in ("embedding", "view_order", "label", "case_id", "coordinates"):
                with self.subTest(corruption=corruption):
                    def corrupted_loader(*args, **kwargs):
                        for batch in original_loader(*args, **kwargs):
                            bags, coords, labels, slides, cases, augmented = batch
                            if corruption == "embedding":
                                bags[0, 0, 0] += 1
                            elif corruption == "view_order":
                                bags[[1, 2]] = bags[[2, 1]]
                            elif corruption == "label":
                                labels[0] = 1 - labels[0]
                            elif corruption == "case_id":
                                cases[0] = "wrong_case"
                            else:
                                coords[0, 0, 0] += 1
                            yield bags, coords, labels, slides, cases, augmented
                    with patch(__name__ + ".create_mil_dataloader", corrupted_loader):
                        with self.assertRaisesRegex(AssertionError,
                                                    "altered|changed|differ"):
                            validate_hdf5_pipeline(path, expected_slides=4)

    def test_raw_padded_bag_is_rejected_before_trimming(self):
        bag = torch.ones(3, 1024)
        bag[-1] = -torch.inf
        coords = torch.tensor([[0, 0], [512, 0], [-1, -1]])
        with self.assertRaisesRegex(ValueError, "NaN|Inf|Padding"):
            AMIL()(bag)
        self.assertEqual(trim_test_padding(bag, coords).shape, (2, 1024))

    def test_trim_rejects_damaged_data_and_misaligned_padding(self):
        for corruption in ("nan", "positive_inf", "partial_negative_inf",
                           "finite_padding", "internal_padding", "bad_coordinate",
                           "partial_coordinate_padding", "all_padding"):
            with self.subTest(corruption=corruption):
                bag = torch.ones(3, 1024)
                bag[-1] = -torch.inf
                coords = torch.tensor([[0, 0], [512, 0], [-1, -1]])
                if corruption == "nan":
                    bag[0, 0] = torch.nan
                elif corruption == "positive_inf":
                    bag[0, 0] = torch.inf
                elif corruption == "partial_negative_inf":
                    bag[0, 0] = -torch.inf
                elif corruption == "finite_padding":
                    bag[-1] = 0
                elif corruption == "internal_padding":
                    bag[[0, 2]] = bag[[2, 0]]
                    coords[[0, 2]] = coords[[2, 0]]
                elif corruption == "bad_coordinate":
                    coords[0, 0] = -1
                elif corruption == "partial_coordinate_padding":
                    coords[-1, 1] = 0
                else:
                    bag[:] = -torch.inf
                    coords[:] = -1
                with self.assertRaises(AssertionError):
                    trim_test_padding(bag, coords)

    def test_source_nonfinite_value_cannot_be_hidden_by_trimming(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupted.h5"
            _write_fixture(path)
            with h5py.File(path, "r+") as target:
                target["slide_00/embeddings"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(AssertionError, "nonfinite embeddings"):
                validate_hdf5_pipeline(path, expected_slides=4)


class AMILPilot14Tests(unittest.TestCase):
    def test_real_pilot14_all_42_full_bags(self):
        path = os.environ.get("PILOT14_H5")
        PILOT14_REPORT.clear()
        if not path:
            PILOT14_REPORT.update(status="not_run", reason="PILOT14_H5 is unset")
            self.skipTest("PILOT14_H5 is unset; real 14-WSI/42-bag acceptance remains incomplete")
        try:
            PILOT14_REPORT["provenance_evidence"] = validate_real_pilot_provenance(path)
            validate_hdf5_pipeline(path, expected_slides=14,
                                   expected_label_counts={0: 7, 1: 7},
                                   report=PILOT14_REPORT)
        except Exception as error:
            PILOT14_REPORT.update(status="fail", error=f"{type(error).__name__}: {error}")
            raise


class AMILAcceptanceCommandTests(unittest.TestCase):
    def test_missing_required_suite_classes_cannot_pass(self):
        from scripts import validate_amil as acceptance
        report = acceptance._suite_coverage(unittest.TestSuite())
        self.assertEqual(report["status"], "fail")
        self.assertEqual(set(report["missing"]), set(acceptance.REQUIRED_TEST_CLASSES))

    def test_skipped_deselected_or_empty_regressions_cannot_pass(self):
        from scripts import validate_amil as acceptance
        for output, expected in (("28 skipped", "fail"),
                                 ("27 passed, 1 skipped", "fail"),
                                 ("27 passed, 1 deselected", "fail"),
                                 ("no tests ran", "fail"), ("28 passed", "pass")):
            with self.subTest(output=output), patch.object(
                    acceptance.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")):
                self.assertEqual(acceptance._regressions()["status"], expected)


if __name__ == "__main__":
    unittest.main()
