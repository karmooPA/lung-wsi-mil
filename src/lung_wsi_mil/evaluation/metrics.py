"""Binary slide-level metrics; never turn undefined AUROC into a numeric zero."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def validate_predictions(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty one-dimensional arrays.")
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have matching lengths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number in [0, 1].")
    return labels.astype(np.int64), probabilities


def binary_metrics(labels, probabilities, *, threshold=0.5):
    """Use Tumor=1; return null plus a reason when a denominator/class is absent."""
    labels, probabilities = validate_predictions(labels, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    undefined = []

    def ratio(numerator, denominator, name):
        if denominator:
            return numerator / denominator
        undefined.append(name)
        return None

    sensitivity = ratio(tp, tp + fn, "sensitivity")
    specificity = ratio(tn, tn + fp, "specificity")
    both_classes = len(np.unique(labels)) == 2
    if not both_classes:
        undefined.extend(["auroc", "average_precision", "balanced_accuracy"])
    return {
        "n_slides": len(labels),
        "positive_slides": int(labels.sum()),
        "negative_slides": int((labels == 0).sum()),
        "positive_prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities))
        if both_classes
        else None,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": ratio(tp, tp + fp, "precision"),
        "negative_predictive_value": ratio(tn, tn + fn, "negative_predictive_value"),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "f1"),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "undefined_metrics": undefined,
        "unit": "one Original view per slide; Tumor is positive",
        "average_precision_definition": "non-interpolated average precision; not trapezoidal PR AUC",
    }


def prediction_curves(labels, probabilities):
    labels, probabilities = validate_predictions(labels, probabilities)
    if len(np.unique(labels)) != 2:
        return {"roc": None, "precision_recall": None, "reason": "Both classes are required."}
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    return {
        "roc": {
            "false_positive_rate": fpr.tolist(),
            "true_positive_rate": tpr.tolist(),
            # sklearn's first ROC threshold is +Inf; JSON has no infinity value.
            "thresholds": [float(value) if np.isfinite(value) else None for value in thresholds],
        },
        "precision_recall": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def patient_bootstrap_auc(labels, probabilities, patient_ids, *, resamples=1000, seed=42):
    """Resample patients with replacement, preserving all their slides together."""
    labels, probabilities = validate_predictions(labels, probabilities)
    patients = np.asarray(patient_ids)
    if patients.shape != labels.shape or any(
        not isinstance(p, str) or not p.strip() for p in patients
    ):
        raise ValueError("One nonempty patient ID is required per slide.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 0:
        raise ValueError("resamples must be a nonnegative integer.")
    unique = sorted(set(patients.tolist()))
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices = np.concatenate(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        )
        if len(np.unique(labels[indices])) == 2:
            values.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    interval = np.quantile(values, [0.025, 0.975]).tolist() if len(values) >= 2 else None
    return {
        "method": "patient-cluster percentile bootstrap",
        "confidence": 0.95,
        "resamples_requested": resamples,
        "resamples_valid": len(values),
        "single_class_resamples_skipped": resamples - len(values),
        "patients": len(unique),
        "seed": int(seed),
        "auroc_interval": interval,
        "reason": None if interval is not None else "At least two valid resamples are required.",
    }


def summarize_seed_metrics(reports):
    """Describe repeated splits without pooling repeated slides as independent."""
    names = (
        "auroc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "brier_score",
        "log_loss",
    )
    summary = {}
    for name in names:
        values = [report[name] for report in reports if report.get(name) is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "defined_seeds": len(values),
            "total_seeds": len(reports),
        }
    return {
        "metrics": summary,
        "interpretation": "Descriptive mean/sample SD across repeated patient splits; seeds are not independent cohorts.",
        "pooled_test_auc": None,
    }
