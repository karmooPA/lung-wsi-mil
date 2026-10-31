"""Kimianet utilities."""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import densenet121

OFFICIAL_SOURCE_COMMIT = "8a2f120999e04e4e18e0057bacbd1b47c27a56be"
OFFICIAL_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/KimiaLabMayo/KimiaNet/"
    f"{OFFICIAL_SOURCE_COMMIT}/KimiaNet_Weights/weights/KimiaNetPyTorchWeights.pth"
)
OFFICIAL_WEIGHTS_SHA256 = "f4972513e308d3197fb940177ca607bb9bb8f82690eadaec1e76190ef8b68375"
OFFICIAL_WEIGHTS_SIZE_BYTES = 28_443_662


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

        # Match the official KimiaNet PyTorch extraction sample at the source
        # commit above: DenseNet features -> adaptive pooling, with no added ReLU.
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


def load_kimianet_weights(model, weights_path, *, expected_sha256=None):
    """Load a flat tensor checkpoint safely and return a frozen eval model.

    Supports the official ``module.model.*`` / ``module.fc_4.*`` names and
    this project's ``features.*`` / ``classifier.*`` names. Optional SHA256
    verification can pin scientific identity independently of the filename.
    The official downloaded weights are for noncommercial research under the
    upstream agreement; keep weights outside the published source repository.
    """
    weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(f"KimiaNet weights not found: {weights_path}")

    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        digest = hashlib.sha256()
        with weights_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError(f"KimiaNet checkpoint SHA256 mismatch: {weights_path}")

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, Mapping):
        raise ValueError("KimiaNet checkpoint must be a flat mapping of tensor names to tensors.")

    converted_state_dict = {}

    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(
                "KimiaNet checkpoint must be a flat mapping of tensor names to tensors."
            )
        if key.startswith("module."):
            key = key[len("module.") :]

        if key.startswith("model."):
            key = "features." + key[len("model.") :]

        elif key.startswith("fc_4."):
            key = "classifier." + key[len("fc_4.") :]

        if key in converted_state_dict:
            raise ValueError(f"Checkpoint keys collide after KimiaNet name conversion: {key}")
        converted_state_dict[key] = value

    # Validate before loading so failed validation does not partly replace a
    # caller's model. Do not silently fill missing tensors or ignore extra ones.
    expected = model.state_dict()
    missing = set(expected).difference(converted_state_dict)
    unexpected = set(converted_state_dict).difference(expected)
    if missing or unexpected:
        raise ValueError(
            "KimiaNet checkpoint keys do not match the model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, value in converted_state_dict.items():
        if value.shape != expected[key].shape:
            raise ValueError(
                f"KimiaNet checkpoint shape mismatch for {key}: "
                f"expected {tuple(expected[key].shape)}, got {tuple(value.shape)}"
            )
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"KimiaNet checkpoint contains non-finite values: {key}")

    model.load_state_dict(
        converted_state_dict,
        strict=True,
    )

    model.requires_grad_(False)
    model.eval()

    return model
