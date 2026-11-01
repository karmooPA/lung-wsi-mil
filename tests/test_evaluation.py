"""Independent metric arithmetic and inference isolation checks."""

import copy

import numpy as np
import pytest
import torch

from lung_wsi_mil.data.mil_collate import mil_collate_fn
from lung_wsi_mil.evaluation.metrics import (
    binary_metrics,
    patient_bootstrap_auc,
    prediction_curves,
    summarize_seed_metrics,
)
from lung_wsi_mil.mil.amil import AMIL
from lung_wsi_mil.mil.evaluation import evaluate_model
from tests.test_training import make_batch


def test_metrics_match_known_rank_and_confusion_arithmetic():
    report = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert report["auroc"] == pytest.approx(0.75)
    assert report["average_precision"] == pytest.approx(5 / 6)
    assert report["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["sensitivity"] == pytest.approx(0.5)
    assert report["specificity"] == pytest.approx(1)
    assert report["f1"] == pytest.approx(2 / 3)
    assert binary_metrics([0, 1], [0.5, 0.5])["confusion_matrix"]["fp"] == 1


def test_single_class_and_no_positive_predictions_are_explicitly_undefined():
    report = binary_metrics([0, 0], [0.1, 0.2])
    assert report["auroc"] is None
    assert report["sensitivity"] is None
    assert report["precision"] is None
    assert "auroc" in report["undefined_metrics"]
    assert prediction_curves([0, 0], [0.1, 0.2])["roc"] is None


@pytest.mark.parametrize(
    "labels,probabilities",
    [([], []), ([0], [float("nan")]), ([2], [0.2]), ([0], [1.01]), ([0, 1], [0.2])],
)
def test_invalid_metric_inputs_are_rejected(labels, probabilities):
    with pytest.raises(ValueError):
        binary_metrics(labels, probabilities)


def test_bootstrap_keeps_a_mixed_label_patients_slides_together():
    result = patient_bootstrap_auc([0, 1], [0.1, 0.9], ["same", "same"], resamples=30, seed=7)
    assert result["resamples_valid"] == 30
    assert result["single_class_resamples_skipped"] == 0
    assert result["auroc_interval"] == [1.0, 1.0]
    assert result == patient_bootstrap_auc(
        [0, 1], [0.1, 0.9], ["same", "same"], resamples=30, seed=7
    )


def test_repeated_seed_summary_does_not_pool_repeated_observations():
    reports = [binary_metrics([0, 1], [0.1, 0.9]), binary_metrics([0, 1], [0.9, 0.1])]
    summary = summarize_seed_metrics(reports)
    assert summary["metrics"]["auroc"]["mean"] == 0.5
    assert summary["metrics"]["auroc"]["sample_std"] == pytest.approx(2**-0.5)
    assert summary["pooled_test_auc"] is None


def _sample(slide, label, count):
    return (
        torch.ones(3, count, 2),
        torch.zeros(count, 2, dtype=torch.long),
        torch.tensor(float(label)),
        slide,
        "patient-" + slide,
    )


def test_original_only_collate_retains_one_bag_and_aligned_metadata_per_slide():
    batch = mil_collate_fn([_sample("a", 0, 2), _sample("b", 1, 1)], view_indices=(0,))
    assert batch[0].shape == (2, 2, 2)
    assert batch[3] == ["a", "b"]
    assert batch[5].tolist() == [False, False]
    assert torch.isneginf(batch[0][1, 1]).all()
    assert (batch[1][1, 1] == -1).all()


def test_evaluation_preserves_parameters_and_training_mode_and_exports_attention():
    torch.manual_seed(7)
    model = AMIL(input_dim=2, attention_dim=3)
    before = copy.deepcopy(model.state_dict())
    batch = make_batch([torch.ones(2, 2), -torch.ones(1, 2)], [1, 0])
    exported = []
    result = evaluate_model(model, [batch], attention_callback=lambda *args: exported.append(args))
    assert model.training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert result["num_slides"] == 2
    assert [item[1].shape for item in exported] == [(2, 2), (1, 2)]
    assert all(np.isclose(item[2].sum(), 1) for item in exported)
    assert all(record["view"] == "Original" for record in result["predictions"])


def test_evaluation_rejects_augmentation_and_misaligned_padding():
    model = AMIL(input_dim=2, attention_dim=3)
    batch = list(make_batch([torch.ones(2, 2), torch.ones(1, 2)], [1, 0]))
    batch[5][0] = True
    with pytest.raises(ValueError, match="Original-only"):
        evaluate_model(model, [batch])
    batch[5][0] = False
    batch[1][1, 1] = 0
    with pytest.raises(ValueError, match="align exactly"):
        evaluate_model(model, [batch])
    assert model.training
