"""Initialization utilities."""

import torch.nn as nn


def initialize_weights(model: nn.Module, method: str = "xavier") -> None:
    if method not in ("xavier", "kaiming"):
        raise ValueError(
            f"Unsupported initialization method: {method}. Expected 'xavier' or 'kaiming'."
        )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif method == "kaiming":
                nn.init.kaiming_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
