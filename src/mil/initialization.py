import torch.nn as nn


def initialize_weights(model: nn.Module, method: str) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier":
                nn.init.xavier_uniform_(module.weight)

            elif method == "kaiming":
                nn.init.kaiming_uniform_(module.weight)

            else:
                raise ValueError(
                    f"Unsupported initialization method: {method}. "
                    "Expected 'xavier' or 'kaiming'."
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)