import torch
import torch.nn as nn

from src.mil.initialization import initialize_weights


class AMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        attention_dim: int = 128,
        k: int = 1,
        threshold: float = 0.5,
        initialization: str = "xavier",
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")

        if attention_dim <= 0:
            raise ValueError("attention_dim must be greater than 0.")

        if k != 1:
            raise ValueError(
                "This AMIL implementation supports only k=1."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.k = k
        self.threshold = threshold
        self.initialization = initialization

        self.attention_hidden = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
        )

        self.attention_score = nn.Linear(
            attention_dim,
            k,
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

        initialize_weights(
            self,
            initialization,
        )

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("AMIL input must be a torch.Tensor.")

        if not x.is_floating_point():
            raise TypeError("AMIL input must have a floating-point dtype.")

        if x.ndim != 2:
            raise ValueError(
                f"Expected input with shape [N, {self.input_dim}], "
                f"but got tensor with shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, "
                f"but got {x.shape[1]}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "AMIL received an empty bag with zero instances."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "AMIL input contains NaN or Inf values. "
                "Padding must be removed before calling the model."
            )

    def _compute_attention_weights(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        attention_hidden = self.attention_hidden(x)

        attention_scores = self.attention_score(
            attention_hidden
        )

        attention_scores = attention_scores.transpose(
            1,
            0,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        return attention_weights

    def _aggregate_bag(
        self,
        x: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        bag_representation = torch.mm(
            attention_weights,
            x,
        )

        return bag_representation

    def forward(
        self,
        x: torch.Tensor,
    ):
        self._validate_input(x)

        attention_weights = self._compute_attention_weights(
            x
        )

        bag_representation = self._aggregate_bag(
            x,
            attention_weights,
        )

        y_prob = self.classifier(
            bag_representation
        )

        y_hat = (
            y_prob >= self.threshold
        ).float()

        return (
            y_prob,
            y_hat,
            attention_weights,
        )
