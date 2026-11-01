"""Publication-exportable ROC/PR and confusion plots without a GUI backend."""

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lung_wsi_mil.evaluation.metrics import prediction_curves


def save_figure(figure, stem, *, formats=("png", "svg")):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    for extension in formats:
        path = stem.with_suffix("." + extension)
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        try:
            figure.savefig(temporary, format=extension, dpi=160, bbox_inches="tight")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    figure.clear()


def plot_seed_metrics(records, metrics, stem, *, title):
    labels = [row["y_true"] for row in records]
    probabilities = [row["probability_tumor"] for row in records]
    curves = prediction_curves(labels, probabilities)
    figure = Figure(figsize=(12, 3.7), layout="constrained")
    roc_axis, pr_axis, matrix_axis = figure.subplots(1, 3)
    if curves["roc"] is not None:
        roc_axis.plot(
            curves["roc"]["false_positive_rate"],
            curves["roc"]["true_positive_rate"],
            color="#166B9D",
            linewidth=2,
            label=f"AUROC {metrics['auroc']:.3f}",
        )
        pr_axis.step(
            curves["precision_recall"]["recall"],
            curves["precision_recall"]["precision"],
            where="post",
            color="#137E69",
            linewidth=2,
            label=f"AP {metrics['average_precision']:.3f}",
        )
        roc_axis.legend(loc="lower right")
        pr_axis.legend(loc="lower left")
    roc_axis.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
    pr_axis.axhline(metrics["positive_prevalence"], linestyle="--", color="0.65", linewidth=1)
    for axis in (roc_axis, pr_axis):
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        axis.grid(alpha=0.18)
    roc_axis.set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC · Original only")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    counts = metrics["confusion_matrix"]
    matrix = np.asarray([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    matrix_axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            matrix_axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    matrix_axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Tumor"],
        yticklabels=["Normal", "Tumor"],
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion · threshold {metrics['threshold']:g}",
    )
    figure.suptitle(title, fontsize=12)
    save_figure(figure, stem)


def plot_repeated_roc(seed_results, stem, *, title):
    figure = Figure(figsize=(6.3, 5.2), layout="constrained")
    axis = figure.subplots()
    for seed, records, metrics in seed_results:
        curves = prediction_curves(
            [row["y_true"] for row in records], [row["probability_tumor"] for row in records]
        )
        if curves["roc"] is not None:
            axis.plot(
                curves["roc"]["false_positive_rate"],
                curves["roc"]["true_positive_rate"],
                linewidth=1.8,
                label=f"Seed {seed}: {metrics['auroc']:.3f}",
            )
    axis.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axis.set(
        xlabel="False-positive rate", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02), title=title
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="lower right", title="Separate held-out splits")
    save_figure(figure, stem)
