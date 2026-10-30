
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize_binary_tensor(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if value.is_complex():
        raise TypeError(f"{name} must not have a complex dtype.")

    if value.ndim == 0:
        value = value.reshape(1)
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value.squeeze(1)
    elif value.ndim != 1:
        raise ValueError(
            f"{name} must be scalar, [B], or [B, 1]; "
            f"got shape {tuple(value.shape)}."
        )

    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")

    return value


def binary_cross_entropy_loss(
    y_prob: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    if not isinstance(y_prob, torch.Tensor):
        raise TypeError("y_prob must be a torch.Tensor.")

    if not y_prob.is_floating_point():
        raise TypeError(
            "y_prob must have a floating-point dtype."
        )

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a real number.")

    if not math.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError("eps must be finite and between 0 and 0.5.")

    probabilities = _normalize_binary_tensor(
        y_prob,
        "y_prob",
    )

    labels = _normalize_binary_tensor(
        targets,
        "targets",
    )

    if probabilities.device != labels.device:
        raise ValueError(
            "y_prob and targets must be on the same device."
        )

    if probabilities.shape != labels.shape:
        raise ValueError(
            "y_prob and targets must contain the same number "
            f"of elements; got {probabilities.numel()} and "
            f"{labels.numel()}."
        )

    if not torch.isfinite(probabilities).all().item():
        raise ValueError("y_prob contains NaN or Inf.")

    if torch.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ).item():
        raise ValueError(
            "y_prob must contain probabilities between 0 and 1."
        )

    labels = labels.to(dtype=probabilities.dtype)

    if not torch.isfinite(labels).all().item():
        raise ValueError("targets contains NaN or Inf.")

    if not torch.all(
        (labels == 0.0) | (labels == 1.0)
    ).item():
        raise ValueError(
            "targets must contain only binary labels 0 and 1."
        )

    probabilities = probabilities.clamp(
        min=eps,
        max=1.0 - eps,
    )

    return F.binary_cross_entropy(
        probabilities,
        labels,
        reduction="mean",
    )
