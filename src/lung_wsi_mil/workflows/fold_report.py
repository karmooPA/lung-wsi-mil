"""Write a relocatable report for one frozen AMIL seed/fold delivery."""

import math
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from lung_wsi_mil.visualization.figures import save_figure

RATE_METRICS = (
    ("auroc", "AUROC"),
    ("average_precision", "Average precision"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
    ("negative_predictive_value", "Negative predictive value"),
    ("f1", "F1"),
    ("brier_score", "Brier score"),
    ("log_loss", "Log loss"),
)

CASES = {
    "true_positive": "True positive (Tumor → Tumor)",
    "true_negative": "True negative (Normal → Normal)",
    "false_positive": "False positive (Normal → Tumor)",
    "false_negative": "False negative (Tumor → Normal)",
}


def _value(value, digits=6):
    return "Not available" if value is None else f"{float(value):.{digits}f}"


def _n(value):
    return "Not available" if value is None else f"{int(value):,}"


def _relative_path(root, value):
    relative = PurePosixPath(str(value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError(f"Expected a path relative to this fold: {value}")
    if not (root / relative).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Report link resolves outside this fold: {value}")
    return relative.as_posix()


def _link(root, label, path):
    relative = _relative_path(root, path)
    return f"[{label}]({quote(relative, safe='/')})" if (root / relative).is_file() else None


def _history_points(history, selected_epoch):
    rows = sorted(history, key=lambda row: int(row["epoch"]))
    epochs = [int(row["epoch"]) for row in rows]
    if not epochs or epochs != list(range(1, len(rows) + 1)) or selected_epoch not in epochs:
        raise ValueError("History must cover consecutive epochs and include the selected epoch.")
    for row in rows:
        if any(
            not math.isfinite(float(row[key]))
            for key in ("train_loss", "validation_loss", "validation_auroc")
        ):
            raise ValueError("History contains a nonfinite training/validation result.")
        if not 0 <= float(row["validation_auroc"]) <= 1:
            raise ValueError("History contains an invalid validation AUROC.")
    return rows, epochs


def _plot_learning_curves(root, rows, epochs, selected_epoch, seed, fold):
    selected = rows[selected_epoch - 1]
    fig = Figure(figsize=(10.4, 3.8), layout="constrained")
    loss_axis, auc_axis = fig.subplots(1, 2)
    for key, color, label in (
        ("train_loss", "#256790", "Training (three views)"),
        ("validation_loss", "#B65D35", "Validation (Original)"),
    ):
        loss_axis.plot(epochs, [row[key] for row in rows], color=color, lw=1.8, label=label)
        loss_axis.scatter(selected_epoch, selected[key], s=24, color=color, zorder=3)
    auc_axis.plot(
        epochs,
        [row["validation_auroc"] for row in rows],
        color="#197867",
        lw=1.8,
        label="Validation AUROC",
    )
    auc_axis.scatter(selected_epoch, selected["validation_auroc"], s=24, color="#197867", zorder=3)
    loss_axis.set(ylabel="Binary cross-entropy", title="Loss by epoch")
    auc_axis.set(ylabel="AUROC", ylim=(0, 1.02), title="Validation ranking performance")
    auc_axis.axhline(0.5, ls=":", lw=1, color="#9BA7AE")
    for axis in (loss_axis, auc_axis):
        axis.axvline(selected_epoch, ls="--", lw=1.15, color="#404953", label="Selected epoch")
        axis.set(xlabel="Epoch", xlim=(0.5, len(epochs) + 0.5))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"Seed {seed} / fold {fold} · {len(epochs)} recorded epochs", fontsize=12)
    save_figure(fig, root / "learning-curves")


def _error_rows(predictions):
    output = []
    for row in predictions:
        label, prediction = int(row["y_true"]), int(row["y_pred"])
        probability = float(row["probability_tumor"])
        if label not in {0, 1} or prediction not in {0, 1} or not 0 <= probability <= 1:
            raise ValueError("A test prediction has invalid class or probability values.")
        if label != prediction:
            output.append((probability if prediction else 1 - probability, row))
    return sorted(output, key=lambda item: (-item[0], item[1]["slide_id"]))


def _confusion_row(metrics):
    matrix = metrics.get("confusion_matrix", {})
    return " | ".join(_n(matrix.get(name)) for name in ("tn", "fp", "fn", "tp"))


def _write_markdown(path, lines):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text("\n".join([*lines, ""]), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_fold_report(
    output,
    *,
    seed,
    fold,
    validation,
    test,
    uncertainty,
    history,
    predictions,
    selected_epoch,
    primary_selected,
    checkpoint_sha256=None,
    cv_loss=None,
    test_loss=None,
    heatmap_index="heatmaps/index.csv",
    metrics_image="metrics.png",
    error_examples=None,
    run_fingerprint=None,
):
    """Summarize saved evidence without fitting, selecting, or evaluating a model.

    All report links are local to the fold and can be opened after moving or
    extracting the delivery. Call this before sealing the containing folder.
    """
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    history, epochs = _history_points(history, selected_epoch)
    predictions = list(predictions)
    errors = _error_rows(predictions)
    confusion = test.get("confusion_matrix", {})
    if len(predictions) != test["n_slides"] or len(errors) != confusion["fp"] + confusion["fn"]:
        raise ValueError("Test predictions do not reconcile with report counts.")
    if len({row["slide_id"] for row in predictions}) != len(predictions):
        raise ValueError("The fold report requires one prediction per test slide.")
    if any(row.get("view", "Original") != "Original" for row in predictions):
        raise ValueError("The fold report requires Original-only test predictions.")
    selected = history[selected_epoch - 1]
    cv_loss = validation.get("loss", selected["validation_loss"]) if cv_loss is None else cv_loss
    _plot_learning_curves(root, history, epochs, selected_epoch, seed, fold)
    interval = uncertainty.get("auroc_interval")
    ci = f"[{_value(interval[0])}, {_value(interval[1])}]" if interval else "Not available"
    confidence = _value(100 * float(uncertainty.get("confidence", 0.95)), 0)
    status = (
        "This is the primary checkpoint originally selected for this seed."
        if primary_selected
        else "This is a supplementary test of a frozen checkpoint outside the primary five."
    )
    lines = [
        f"# AMIL results for seed {seed}, fold {fold}",
        "",
        status,
        "",
        f"The held-out test used **{_n(test['n_slides'])} slides** from "
        f"**{_n(test.get('n_patients'))} patients**. Test AUROC was "
        f"**{_value(test.get('auroc'))}** ({confidence}% patient-cluster bootstrap CI {ci}). "
        f"Accuracy was **{_value(test.get('accuracy'))}** at the frozen threshold "
        f"**{_value(test.get('threshold'), 2)}**.",
        "",
        "## Validation and held-out test",
        "",
        "| Metric | Inner validation | Outer held-out test |",
        "| --- | ---: | ---: |",
        f"| Slides | {_n(validation.get('n_slides'))} | {_n(test.get('n_slides'))} |",
        f"| Normal slides | {_n(validation.get('negative_slides'))} | "
        f"{_n(test.get('negative_slides'))} |",
        f"| Tumor slides | {_n(validation.get('positive_slides'))} | "
        f"{_n(test.get('positive_slides'))} |",
    ]
    lines += [
        f"| {label} | {_value(validation.get(key))} | {_value(test.get(key))} |"
        for key, label in RATE_METRICS
    ]
    lines += [
        f"| Clipped probability BCE | {_value(cv_loss)} | {_value(test_loss)} |",
        "",
        "| Partition | True negatives | False positives | False negatives | True positives |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Validation | {_confusion_row(validation)} |",
        f"| Test | {_confusion_row(test)} |",
        "",
        f"Bootstrap used {_n(uncertainty.get('resamples_valid'))} valid draws out of "
        f"{_n(uncertainty.get('resamples_requested'))}. Each draw kept all slides from each "
        "sampled patient together. The interval is for slide-level test AUROC. Average precision "
        "is the non-interpolated AP metric.",
        "",
    ]
    if validation.get("auroc") is not None and test.get("auroc") is not None:
        delta = test["auroc"] - validation["auroc"]
        lines += [
            f"The test minus validation AUROC difference is **{delta:+.6f}**. These are "
            "different patient sets, and validation selected the checkpoint. The difference "
            "is descriptive.",
            "",
        ]
    lines += [
        "## Training history",
        "",
        f"Epoch **{selected_epoch}** of **{len(epochs)}** recorded epochs was selected on "
        "validation AUROC, using validation BCE to break exact ties. At that epoch the training "
        f"BCE was {_value(selected['train_loss'])}, validation BCE {_value(cv_loss)}, and "
        f"validation AUROC {_value(selected['validation_auroc'])}. Training used the Original, "
        "Aug1 and Aug2 bags. Validation and test each used one Original bag per slide.",
        "",
        "![Training and validation curves](learning-curves.png)",
        "",
        "[Learning curves as SVG](learning-curves.svg)",
        "",
    ]
    metrics_link = _link(root, "Test ROC, precision–recall and confusion figure", metrics_image)
    if metrics_link:
        metrics_path = _relative_path(root, metrics_image)
        lines += [
            "## Test curves and confusion",
            "",
            f"![ROC, precision–recall and confusion]({quote(metrics_path, safe='/')})",
            "",
        ]
        metrics_svg = _link(
            root, "Test curves and confusion as SVG", Path(metrics_path).with_suffix(".svg")
        )
        if metrics_svg:
            lines += [metrics_svg, ""]
    lines += [
        "## Test errors and attention",
        "",
        f"The model made **{_n(len(errors))} errors**, including {_n(confusion['fp'])} false "
        f"positives and {_n(confusion['fn'])} false negatives. The table lists up to five errors "
        "ranked by probability assigned to the predicted class, with slide ID breaking ties.",
        "",
    ]
    if errors:
        lines += [
            "| Slide | True label | Prediction | Tumor probability | Wrong-class probability |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for confidence_score, row in errors[:5]:
            true_label = "Tumor" if int(row["y_true"]) else "Normal"
            predicted = "Tumor" if int(row["y_pred"]) else "Normal"
            slide = str(row["slide_id"]).replace("|", "\\|")
            lines.append(
                f"| {slide} | {true_label} | {predicted} | {_value(row['probability_tumor'])} "
                f"| {_value(confidence_score)} |"
            )
        lines.append("")
    heatmap_link = _link(root, "Index of all test-slide heatmaps", heatmap_index)
    if heatmap_link:
        lines += [heatmap_link, ""]
    for key, label in CASES.items():
        path = (error_examples or {}).get(key)
        if path:
            example = _link(root, label, path)
            if example:
                lines.append(f"- {example}")
    lines += [
        "",
        "Attention colors represent relative patch ranks within a slide. Areas outside the "
        "sampled patches appear transparent.",
        "",
        "## Protocol and files",
        "",
        "The five folds of this seed share the same outer test patients, so their test results "
        "are correlated. Model and epoch selection used validation outcomes, and the decision "
        "threshold was fixed before test evaluation.",
        "",
    ]
    if run_fingerprint:
        lines.append(f"Frozen experiment fingerprint `{run_fingerprint}`.")
    if checkpoint_sha256:
        lines.append(f"Selected checkpoint SHA256 `{checkpoint_sha256}`.")
    lines.append("")
    for label, path in (
        ("Selected checkpoint", "train/best.pt"),
        ("Epoch history", "train/history.json"),
        ("Validation predictions", "train/validation_predictions.csv"),
        ("Validation metrics", "train/validation_metrics.json"),
        ("Test predictions", "test/predictions.csv"),
        ("Test metrics", "test/metrics.json"),
        ("Patient bootstrap data", "test/patient_bootstrap.json"),
        ("ROC and PR curve points", "test/curves.json"),
    ):
        artifact = _link(root, label, path)
        if artifact:
            lines.append(f"- {artifact}")
    _write_markdown(root / "report.md", lines)
    return {
        "report": "report.md",
        "learning_curves": ["learning-curves.png", "learning-curves.svg"],
        "test_slides": len(predictions),
        "test_errors": len(errors),
    }
