
from __future__ import annotations

import torch


def remove_padding(bag_embeddings: torch.Tensor) -> torch.Tensor:
    if not isinstance(bag_embeddings, torch.Tensor):
        raise TypeError("bag_embeddings must be a torch.Tensor.")

    if not bag_embeddings.is_floating_point():
        raise TypeError(
            "bag_embeddings must have a floating-point dtype."
        )

    if bag_embeddings.ndim != 2:
        raise ValueError(
            "bag_embeddings must have shape [N, feature_dim], "
            f"got {tuple(bag_embeddings.shape)}."
        )

    if bag_embeddings.shape[0] == 0:
        raise ValueError(
            "bag_embeddings must contain at least one instance."
        )

    if bag_embeddings.shape[1] == 0:
        raise ValueError(
            "bag_embeddings must contain at least one feature."
        )

    padding_mask = torch.isneginf(bag_embeddings).all(dim=1)
    valid_mask = ~padding_mask

    if not valid_mask.any().item():
        raise ValueError(
            "The bag contains no valid instances."
        )

    valid_embeddings = bag_embeddings[valid_mask]

    if not torch.isfinite(valid_embeddings).all().item():
        raise ValueError(
            "Non-padding instances must contain only finite values."
        )

    return valid_embeddings
