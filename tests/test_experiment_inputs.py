"""Reject leakage, changed identities, incomplete caches and invalid protocols."""

import hashlib
import json
from dataclasses import replace

import h5py
import pandas as pd
import pytest

from lung_wsi_mil.experiments.artifacts import fingerprint, hash_file
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.fold import is_improved, meaningful_progress
from lung_wsi_mil.experiments.inputs import load_metadata, validate_features, validate_partitions
from lung_wsi_mil.experiments.pipeline import select_final_fold
from lung_wsi_mil.features.kimianet import OFFICIAL_SOURCE_COMMIT, OFFICIAL_WEIGHTS_SHA256
from tests.experiment_fixtures import make_experiment_inputs


@pytest.fixture
def inputs(tmp_path):
    return make_experiment_inputs(tmp_path / "inputs")


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [2, 2]),
        ("epochs", 0),
        ("folds", 4),
        ("learning_rate", -1),
        ("threshold", 1.1),
        ("patience", True),
        ("heatmap_max_size", 5000),
        ("bootstrap_resamples", -1),
    ],
)
def test_invalid_protocol_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{field: value})


def test_unknown_configuration_keys_are_not_silently_ignored():
    with pytest.raises(ValueError, match="Unknown"):
        ExperimentConfig.from_dict({"epoch": 3})


def test_all_five_seed_partitions_are_valid_and_frozen(inputs):
    metadata = load_metadata(inputs["inventory"])
    partitions, hashes = validate_partitions(metadata, inputs["splits"], ExperimentConfig())
    assert len(partitions) == 5 and len(hashes) == 60
    for frames in partitions.values():
        assert not set(frames["train_cv"].patient_id) & set(frames["test"].patient_id)
        assert set(frames["fold_1_val"].label) == {0, 1}


def test_split_label_corruption_is_rejected_before_model_training(inputs):
    path = inputs["splits"] / "seed_2/fold_1_val.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "label"] = "Normal" if frame.loc[0, "label"] == "Tumor" else "Tumor"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicts"):
        validate_partitions(
            load_metadata(inputs["inventory"]), inputs["splits"], ExperimentConfig()
        )


def test_cache_without_provenance_requires_explicit_unverified_override(inputs):
    metadata = load_metadata(inputs["inventory"])
    kwargs = {"inventory_sha256": hash_file(inputs["inventory"])}
    with pytest.raises(ValueError, match="provenance"):
        validate_features(inputs["features"], metadata, **kwargs)
    result = validate_features(inputs["features"], metadata, allow_unverified=True, **kwargs)
    assert not result["official_provenance_verified"]
    assert result["input_scope"] == "SYNTHETIC_OR_UNVERIFIED"


def test_nonfinite_features_and_missing_slides_are_explicit_failures(inputs):
    metadata = load_metadata(inputs["inventory"])
    with h5py.File(inputs["features"], "a") as file:
        first = next(iter(file))
        file[first]["embeddings"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_features(
            inputs["features"], metadata, inventory_sha256="unused", allow_unverified=True
        )
    with h5py.File(inputs["features"], "a") as file:
        del file[first]
    with pytest.raises(ValueError, match="exactly the inventory"):
        validate_features(
            inputs["features"], metadata, inventory_sha256="unused", allow_unverified=True
        )


def test_final_fold_selection_uses_only_validation_and_deterministic_ties():
    summaries = [
        {"fold": 1, "validation": {"auroc": 0.8, "loss": 0.4}, "test_auc": 1.0},
        {"fold": 2, "validation": {"auroc": 0.9, "loss": 0.5}, "test_auc": 0.0},
        {"fold": 3, "validation": {"auroc": 0.9, "loss": 0.6}, "test_auc": 1.0},
    ]
    assert select_final_fold(summaries)["fold"] == 2
    summaries[2]["validation"]["loss"] = 0.5
    assert select_final_fold(list(reversed(summaries)))["fold"] == 2


def test_checkpoint_rank_never_trades_a_lower_auc_for_a_better_loss():
    best = {"auroc": 0.9, "loss": 0.5}
    assert not is_improved({"auroc": 0.89999, "loss": 0.1}, best)
    assert is_improved({"auroc": 0.90001, "loss": 0.9}, best)
    assert is_improved({"auroc": 0.9, "loss": 0.49999}, best)
    assert not meaningful_progress({"auroc": 0.90001, "loss": 0.9}, best, 0.001)


def test_verified_format_checks_every_embedding_checksum(inputs):
    # This fabricates metadata only to test the validator's format contract;
    # no official encoder execution or real-data result is asserted by this fixture.
    inventory_hash = hash_file(inputs["inventory"])
    metadata = load_metadata(inputs["inventory"])
    sources = []
    with h5py.File(inputs["features"], "a") as file:
        for row in metadata.itertuples():
            group = file[row.slide_id]
            source = {
                "slide_id": row.slide_id,
                "case_id": row.patient_id,
                "source_label": "Tumor" if row.label else "Normal",
                "label": row.label,
                "n": group["embeddings"].shape[1],
                "sha256": "b" * 64,
                "run_fingerprint": "synthetic-validator-fixture",
            }
            sources.append(source)
            group.attrs["source_sha256"] = source["sha256"]
            group.attrs["source_run_fingerprint"] = source["run_fingerprint"]
            group.attrs["embeddings_sha256"] = hashlib.sha256(
                group["embeddings"][:].tobytes()
            ).hexdigest()
            group.attrs["coords_sha256"] = hashlib.sha256(group["coords"][:].tobytes()).hexdigest()
        provenance = {
            "config": {"fixture": True},
            "code": {"fixture": True},
            "weights_sha256": OFFICIAL_WEIGHTS_SHA256,
            "encoder_source_commit": OFFICIAL_SOURCE_COMMIT,
            "inventory_sha256": inventory_hash,
            "sources": sources,
        }
        file.attrs.update(
            status="complete",
            format_version="1.0",
            provenance_json=json.dumps(provenance),
            fingerprint=fingerprint(provenance),
            config_sha256=fingerprint(provenance["config"]),
            code_sha256=fingerprint(provenance["code"]),
            weights_sha256=OFFICIAL_WEIGHTS_SHA256,
        )
    report = validate_features(inputs["features"], metadata, inventory_sha256=inventory_hash)
    assert report["official_provenance_verified"]
    with h5py.File(inputs["features"], "a") as file:
        file[sources[0]["slide_id"]]["embeddings"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_features(inputs["features"], metadata, inventory_sha256=inventory_hash)
