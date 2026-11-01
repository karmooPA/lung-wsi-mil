"""Real optimizer/checkpoint paths on synthetic data, including an interrupted epoch."""

import json
from dataclasses import replace

import pandas as pd
import pytest
import torch

from lung_wsi_mil.experiments import fold as fold_module
from lung_wsi_mil.experiments.artifacts import fingerprint, hash_file, load_checkpoint, read_json
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.fold import fit_fold
from lung_wsi_mil.experiments.inputs import load_metadata, validate_partitions
from lung_wsi_mil.experiments.pipeline import evaluate_experiment, train_experiment
from lung_wsi_mil.experiments.reproducibility import configure_runtime
from tests.experiment_fixtures import make_experiment_inputs


def _config(**kwargs):
    return ExperimentConfig(
        epochs=1,
        patience=2,
        attention_dim=4,
        batch_size=4,
        bootstrap_resamples=10,
        heatmap_limit=1,
        heatmap_max_size=128,
        allow_unverified_features=True,
        **kwargs,
    )


def test_complete_25_fit_workflow_selects_before_test_and_exports_spatial_maps(
    tmp_path, monkeypatch
):
    inputs = make_experiment_inputs(tmp_path / "inputs")
    output = tmp_path / "experiment"
    config = _config()
    trained = train_experiment(
        features=inputs["features"],
        inventory=inputs["inventory"],
        splits=inputs["splits"],
        output=output,
        config=config,
    )
    assert trained["completed_fits"] == 25
    assert trained["test_forward_passes"] == 0
    assert not (output / "test").exists()
    assert len(list((output / "training").glob("seed_*/fold_*/best.pt"))) == 25
    for selection in trained["seeds"]:
        seed = selection["seed"]
        oof = pd.read_csv(output / f"training/seed_{seed}/validation_oof_predictions.csv")
        test = pd.read_csv(inputs["splits"] / f"seed_{seed}/test.csv")
        assert not set(oof.patient_id) & set(test.patient_id)
        assert not oof.slide_id.duplicated().any()

    def forbidden_training(*args, **kwargs):
        raise AssertionError("Final test must never optimize a model.")

    monkeypatch.setattr(fold_module, "train_one_epoch", forbidden_training)
    result = evaluate_experiment(output=output, patch_root=inputs["patch_root"])
    assert result["completed_test_seeds"] == 5
    assert result["completed_training_fits"] == 25
    assert result["attention_slides"] == 20
    assert result["heatmaps"] == 5
    assert result["input_scope"] == "SYNTHETIC_OR_UNVERIFIED"
    assert not result["verified_reference_cohort"]
    assert result["aggregate"]["pooled_test_auc"] is None
    assert len(list((output / "test").glob("seed_*/heatmaps/*/overlay.png"))) == 5
    assert len(list((output / "test").glob("seed_*/attention/*.csv"))) == 20
    for selection, tested in zip(trained["seeds"], result["seeds"], strict=True):
        assert selection["selected_fold"] == tested["selected_fold"]
        assert selection["checkpoint_sha256"] == tested["checkpoint_sha256"]
        assert tested["metrics"]["n_slides"] == 4
        assert tested["metrics"]["n_patients"] == 2
        assert not tested["test_used_for_selection"]
    with pytest.raises(FileExistsError, match="Final-test output exists"):
        evaluate_experiment(output=output, patch_root=inputs["patch_root"])
    import lung_wsi_mil.mil.evaluation as evaluation

    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda *a, **kw: pytest.fail("Cached test must not forward again."),
    )
    assert (
        evaluate_experiment(output=output, patch_root=inputs["patch_root"], resume=True) == result
    )


def test_interrupted_fold_resume_matches_uninterrupted_model_and_history(tmp_path, monkeypatch):
    inputs = make_experiment_inputs(tmp_path / "inputs")
    config = replace(_config(seeds=(2,)), epochs=3)
    configure_runtime(config)
    frames, hashes = validate_partitions(
        load_metadata(inputs["inventory"]), inputs["splits"], config
    )
    context = {
        "fingerprint": "a" * 64,
        "config_hash": fingerprint(config.to_dict()),
        "inventory_sha256": hash_file(inputs["inventory"]),
        "split_hash": fingerprint(hashes),
        "features": {"sha256": hash_file(inputs["features"])},
    }
    kwargs = {
        "features": inputs["features"],
        "train_frame": frames[2]["fold_1_train"],
        "val_frame": frames[2]["fold_1_val"],
        "config": config,
        "context": context,
        "seed": 2,
        "fold": 1,
    }
    baseline = fit_fold(directory=tmp_path / "baseline", **kwargs)
    original = fold_module.train_one_epoch
    calls = 0

    def interrupted(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption after an atomic epoch")
        return original(*args, **kw)

    monkeypatch.setattr(fold_module, "train_one_epoch", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit_fold(directory=tmp_path / "resumed", **kwargs)
    assert (tmp_path / "resumed/last.pt").is_file()
    assert not (tmp_path / "resumed/complete.json").exists()
    monkeypatch.setattr(fold_module, "train_one_epoch", original)
    resumed = fit_fold(directory=tmp_path / "resumed", **kwargs)
    assert baseline["validation"] == resumed["validation"]
    assert read_json(tmp_path / "baseline/history.json") == read_json(
        tmp_path / "resumed/history.json"
    )
    states = [
        load_checkpoint(tmp_path / name / "best.pt", run_fingerprint="a" * 64, seed=2, fold=1)
        for name in ("baseline", "resumed")
    ]
    assert all(
        torch.equal(states[0]["model_state"][key], states[1]["model_state"][key])
        for key in states[0]["model_state"]
    )


def test_resume_rejects_changed_protocol_and_corrupt_completed_checkpoints(tmp_path):
    inputs = make_experiment_inputs(tmp_path / "inputs")
    config = _config(seeds=(2,))
    kwargs = {
        "features": inputs["features"],
        "inventory": inputs["inventory"],
        "splits": inputs["splits"],
        "output": tmp_path / "experiment",
    }
    train_experiment(config=config, **kwargs)
    with pytest.raises(ValueError, match="Resume refused"):
        train_experiment(config=replace(config, epochs=2), resume=True, **kwargs)
    manifest_path = kwargs["output"] / "run.json"
    original_manifest = manifest_path.read_text()
    changed = json.loads(original_manifest)
    changed["features"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="input references differ"):
        evaluate_experiment(output=kwargs["output"])
    manifest_path.write_text(original_manifest)
    checkpoint = kwargs["output"] / "training/seed_2/fold_1/best.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="corrupt"):
        train_experiment(config=config, resume=True, **kwargs)


def test_invalid_cache_creates_no_experiment_or_fake_results(tmp_path):
    inputs = make_experiment_inputs(tmp_path / "inputs")
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match="provenance"):
        train_experiment(
            features=inputs["features"],
            inventory=inputs["inventory"],
            splits=inputs["splits"],
            output=output,
            config=replace(_config(), allow_unverified_features=False),
        )
    assert not output.exists()
