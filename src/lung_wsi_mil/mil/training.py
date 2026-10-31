"""Train padded MIL batches one finite, unpadded view-bag at a time."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from lung_wsi_mil.mil.bag_utils import remove_padding
from lung_wsi_mil.mil.losses import binary_cross_entropy_loss


def _resolve_device(
    model: nn.Module,
    device: str | torch.device | None,
) -> torch.device:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")

    parameters = list(model.parameters())

    if not parameters:
        raise ValueError("model must contain parameters.")

    model_devices = {parameter.device for parameter in parameters}

    if len(model_devices) != 1:
        raise ValueError("All model parameters must be on the same device.")

    model_device = next(iter(model_devices))
    target_device = model_device if device is None else torch.device(device)

    # "cuda" means the model's current CUDA index, not a different device.
    if target_device.type == model_device.type and target_device.index is None:
        target_device = model_device

    if target_device != model_device:
        raise ValueError("Move the model to the requested device before training.")

    return target_device


def _normalize_labels(
    labels: torch.Tensor,
    num_bags: int,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor.")

    if labels.is_complex():
        raise TypeError("labels must not have a complex dtype.")

    if labels.ndim == 0:
        labels = labels.reshape(1)
    elif labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    elif labels.ndim != 1:
        raise ValueError(f"labels must have shape [B] or [B, 1]; got {tuple(labels.shape)}.")

    if labels.numel() != num_bags:
        raise ValueError(f"Expected {num_bags} labels, got {labels.numel()}.")

    if not torch.isfinite(labels).all().item():
        raise ValueError("labels contain NaN or Inf.")

    if not torch.all((labels == 0) | (labels == 1)).item():
        raise ValueError("labels must contain only 0 and 1.")

    return labels


def _prepare_batch(
    batch: Any,
    model: nn.Module,
    device: torch.device,
):
    if not isinstance(batch, (tuple, list)) or len(batch) != 6:
        raise ValueError(
            "Expected the six outputs of mil_collate_fn: "
            "padded_bags, coordinates, labels, slide_ids, "
            "case_ids, is_augmented."
        )

    (
        padded_bags,
        coordinates,
        labels,
        slide_ids,
        case_ids,
        is_augmented,
    ) = batch

    if not isinstance(padded_bags, torch.Tensor):
        raise TypeError("padded_bags must be a torch.Tensor.")

    if not padded_bags.is_floating_point():
        raise TypeError("padded_bags must have a floating-point dtype.")

    if padded_bags.ndim != 3:
        raise ValueError(f"padded_bags must have shape [B, N, D]; got {tuple(padded_bags.shape)}.")

    num_bags, max_instances, feature_dim = padded_bags.shape

    if num_bags == 0 or max_instances == 0 or feature_dim == 0:
        raise ValueError("padded_bags must have non-zero dimensions.")

    expected_dim = getattr(model, "input_dim", None)

    if expected_dim is not None and feature_dim != expected_dim:
        raise ValueError(f"Expected feature dimension {expected_dim}, got {feature_dim}.")

    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a torch.Tensor.")

    if (
        coordinates.ndim != 3
        or coordinates.shape[0] != num_bags
        or coordinates.shape[1] != max_instances
        or coordinates.shape[2] != 2
    ):
        raise ValueError("coordinates must have shape [B, N, 2] matching padded_bags.")

    if not isinstance(slide_ids, (tuple, list)):
        raise TypeError("slide_ids must be a sequence.")

    if not isinstance(case_ids, (tuple, list)):
        raise TypeError("case_ids must be a sequence.")

    if len(slide_ids) != num_bags or len(case_ids) != num_bags:
        raise ValueError("slide_ids and case_ids must contain one entry per bag.")

    if not all(isinstance(slide_id, str) for slide_id in slide_ids):
        raise TypeError("Every slide_id must be a string.")

    if not all(isinstance(case_id, str) for case_id in case_ids):
        raise TypeError("Every case_id must be a string.")

    if not isinstance(is_augmented, torch.Tensor):
        raise TypeError("is_augmented must be a torch.Tensor.")

    if is_augmented.dtype != torch.bool:
        raise TypeError("is_augmented must have dtype torch.bool.")

    if is_augmented.shape != (num_bags,):
        raise ValueError("is_augmented must have shape [B] matching padded_bags.")

    labels = _normalize_labels(labels, num_bags)

    padded_bags = padded_bags.to(device=device)
    labels = labels.to(device=device)

    return padded_bags, labels, slide_ids


def train_batch(
    model: nn.Module,
    optimizer: Optimizer,
    batch: Any,
    device: str | torch.device | None = None,
    eps: float = 1e-5,
) -> dict[str, float | int]:
    """Optimize one padded batch and report the mean loss over its view-bags."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")

    target_device = _resolve_device(model, device)

    (
        padded_bags,
        labels,
        slide_ids,
    ) = _prepare_batch(
        batch,
        model,
        target_device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    probabilities = []

    for bag_index in range(padded_bags.shape[0]):
        bag = remove_padding(padded_bags[bag_index])

        output = model(bag)

        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise ValueError("AMIL must return (y_prob, y_hat, attention_weights).")

        y_prob = output[0]

        if not isinstance(y_prob, torch.Tensor):
            raise TypeError("AMIL y_prob must be a torch.Tensor.")

        if not y_prob.is_floating_point():
            raise TypeError("AMIL y_prob must have a floating-point dtype.")

        if y_prob.numel() != 1:
            raise ValueError("AMIL must return exactly one probability per bag.")

        if y_prob.device != target_device:
            raise ValueError("AMIL output and model parameters must use the same device.")

        probabilities.append(y_prob.reshape(1))

    y_prob = torch.cat(probabilities, dim=0)

    loss = binary_cross_entropy_loss(
        y_prob=y_prob,
        targets=labels,
        eps=eps,
    )

    if not torch.isfinite(loss).item():
        raise FloatingPointError("Training loss is not finite.")

    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "num_bags": int(padded_bags.shape[0]),
        "num_slides": int(len(set(slide_ids))),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    optimizer: Optimizer,
    device: str | torch.device | None = None,
    eps: float = 1e-5,
) -> dict[str, float | int]:
    """Train all batches, weighting loss by bag count and counting unique slides."""
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")

    if not hasattr(dataloader, "__iter__"):
        raise TypeError("dataloader must be iterable.")

    _resolve_device(model, device)
    model.train()

    total_loss = 0.0
    total_bags = 0
    total_batches = 0
    seen_slides: set[str] = set()

    for batch in dataloader:
        batch_result = train_batch(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            eps=eps,
        )

        batch_bags = int(batch_result["num_bags"])
        batch_loss = float(batch_result["loss"])

        total_loss += batch_loss * batch_bags
        total_bags += batch_bags
        total_batches += 1
        seen_slides.update(batch[3])

    if total_batches == 0 or total_bags == 0:
        raise ValueError("The training dataloader produced no bags.")

    return {
        "loss": total_loss / total_bags,
        "num_batches": total_batches,
        "num_bags": total_bags,
        "num_slides": len(seen_slides),
    }
