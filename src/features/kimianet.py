from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import densenet121


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class KimiaNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        densenet = densenet121(weights=None)

        self.features = nn.Sequential(
            densenet.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.feature_dim = densenet.classifier.in_features
        self.classifier = nn.Linear(self.feature_dim, 30)

    def forward(self, x):
        x = self.features(x)

        embedding = torch.flatten(x, 1)
        logits = self.classifier(embedding)

        return embedding, logits


def load_kimianet_weights(model, weights_path):
    weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"KimiaNet weights not found: {weights_path}"
        )

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
    )

    converted_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("model."):
            key = "features." + key[len("model."):]

        elif key.startswith("fc_4."):
            key = "classifier." + key[len("fc_4."):]

        converted_state_dict[key] = value

    model.load_state_dict(
        converted_state_dict,
        strict=True,
    )

    model.requires_grad_(False)
    model.eval()

    return model