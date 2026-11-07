"""A frozen fold delivers a report and canonical overlays for every test slide."""

import numpy as np
from PIL import Image

from lung_wsi_mil.data.mil_hdf5_dataset import MILHDF5Dataset
from lung_wsi_mil.experiments.artifacts import verify_stage
from lung_wsi_mil.experiments.config import ExperimentConfig
from lung_wsi_mil.experiments.pipeline import load_frozen_run, train_experiment
from lung_wsi_mil.experiments.synthetic import make_synthetic_inputs
from lung_wsi_mil.visualization.attention import render_from_preprocessing
from lung_wsi_mil.workflows.fold_delivery import _evaluate_pair, _geometry_inventory


def test_fold_delivery_keeps_every_original_overlay_and_report(tmp_path):
    inputs = make_synthetic_inputs(tmp_path / "inputs")
    source = tmp_path / "trained"
    config = ExperimentConfig(
        seeds=(2,),
        epochs=1,
        patience=1,
        attention_dim=4,
        bootstrap_resamples=3,
        allow_unverified_features=True,
    )
    train_experiment(
        features=inputs["features"],
        inventory=inputs["inventory"],
        splits=inputs["splits"],
        output=source,
        config=config,
    )
    _, features, config, manifest, training, partitions, _ = load_frozen_run(source)
    pair = tmp_path / "pair"
    pair.mkdir()
    patch_root = inputs["patch_root"]
    result = _evaluate_pair(
        pair=pair,
        seed=2,
        fold=1,
        train_source=source / "training/seed_2/fold_1",
        selection=training["seeds"][0],
        dataset=MILHDF5Dataset(features),
        frame=partitions[2]["test"],
        config=config,
        manifest=manifest,
        patch_root=patch_root,
        geometry=_geometry_inventory(patch_root, partitions, features),
        render_workers=2,
        max_size=128,
        delivery_id="synthetic-test-id",
    )
    assert result["heatmaps"] == len(partitions[2]["test"])
    assert verify_stage(pair, run_fingerprint="synthetic-test-id")["heatmaps"] == result["heatmaps"]
    assert (pair / "train/validation_predictions.csv").is_file()
    assert (pair / "learning-curves.png").is_file()
    assert "test/heatmaps/index.csv" in (pair / "report.md").read_text(encoding="utf-8")
    for slide in partitions[2]["test"].slide_id:
        baseline = tmp_path / "baseline" / slide
        render_from_preprocessing(
            pair / "test/attention" / f"{slide}.csv",
            patch_root / "slides" / slide / "preprocessing",
            baseline,
            max_size=128,
        )
        for name in ("heatmap.png", "overlay.png", "coverage.png"):
            with (
                Image.open(pair / "test/heatmaps" / slide / name) as optimized,
                Image.open(baseline / name) as original,
            ):
                assert np.array_equal(np.asarray(optimized), np.asarray(original)), (slide, name)
