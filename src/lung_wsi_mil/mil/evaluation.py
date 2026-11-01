"""Original-only inference and validation with exact padding/coordinate alignment."""

import numpy as np
import torch

from lung_wsi_mil.evaluation.metrics import binary_metrics
from lung_wsi_mil.mil.losses import binary_cross_entropy_loss
from lung_wsi_mil.mil.training import _prepare_batch, _resolve_device


def evaluate_model(model, dataloader, *, device=None, threshold=0.5, attention_callback=None):
    """Evaluate one unaugmented bag per slide without changing model parameters.

    A callback receives ``(prediction_dict, coordinates, weights)`` while that
    bag is in memory. Arrays are not retained in the returned scalar records.
    """
    target_device = _resolve_device(model, device)
    prior_training = model.training
    records, seen = [], set()
    model.eval()
    try:
        with torch.inference_mode():
            for batch in dataloader:
                bags, labels, slide_ids = _prepare_batch(batch, model, target_device)
                coordinates, patient_ids, augmented = batch[1], batch[4], batch[5]
                if bool(augmented.any()):
                    raise ValueError(
                        "Validation/test must use an Original-only loader (view_indices=(0,))."
                    )
                if coordinates.dtype != torch.int64:
                    raise ValueError("Coordinates must be int64.")
                for index, slide_id in enumerate(slide_ids):
                    if slide_id in seen:
                        raise ValueError(f"Duplicate evaluation slide: {slide_id}")
                    seen.add(slide_id)
                    bag, coords = bags[index], coordinates[index]
                    padding = torch.isneginf(bag).all(dim=1)
                    mask = padding.cpu()
                    count = int((~mask).sum())
                    if not count or bool(mask[:count].any()) or bool((~mask[count:]).any()):
                        raise ValueError(
                            "Padding must be a complete contiguous tail after a nonempty bag."
                        )
                    if not bool(torch.all(coords[mask] == -1)) or bool((coords[~mask] < 0).any()):
                        raise ValueError(
                            "Coordinate padding must align exactly with embedding padding."
                        )
                    bag = bag[:count]
                    if not bool(torch.isfinite(bag).all()):
                        raise ValueError("Unpadded embeddings must be finite.")
                    probability, prediction, attention = model(bag)
                    if probability.numel() != 1 or not bool(torch.isfinite(probability).all()):
                        raise ValueError("Model must return one finite probability per bag.")
                    probability = float(probability.item())
                    if not 0 <= probability <= 1:
                        raise ValueError("Model probability must be in [0,1].")
                    if (
                        prediction.numel() != 1
                        or float(prediction.item()) not in (0.0, 1.0)
                        or float(prediction.item()) != int(probability >= threshold)
                    ):
                        raise ValueError("Model prediction does not match the frozen threshold.")
                    weights = attention.detach().cpu().numpy()
                    if (
                        weights.shape != (1, count)
                        or not np.isfinite(weights).all()
                        or (weights < 0).any()
                        or not np.isclose(weights.sum(), 1, atol=1e-5)
                    ):
                        raise ValueError(
                            "Attention must be finite, nonnegative, [1,N] and sum to one."
                        )
                    record = {
                        "slide_id": slide_id,
                        "patient_id": patient_ids[index],
                        "y_true": int(labels[index].item()),
                        "probability_tumor": probability,
                        "threshold": float(threshold),
                        "y_pred": int(probability >= threshold),
                        "view": "Original",
                        "patch_count": count,
                    }
                    records.append(record)
                    if attention_callback is not None:
                        attention_callback(record, coords[:count].cpu().numpy(), weights[0])
    finally:
        model.train(prior_training)
    if not records:
        raise ValueError("Evaluation loader produced no slides.")
    probabilities = [record["probability_tumor"] for record in records]
    labels = [record["y_true"] for record in records]
    loss = binary_cross_entropy_loss(
        torch.tensor(probabilities, dtype=torch.float64), torch.tensor(labels, dtype=torch.float64)
    )
    return {
        "loss": float(loss.item()),
        "num_slides": len(records),
        "predictions": records,
        "metrics": binary_metrics(labels, probabilities, threshold=threshold),
    }
