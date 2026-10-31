"""Checkpoint routing/safety checks; no trained weights are redistributed."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lung_wsi_mil.features.kimianet import load_kimianet_weights


class TinyEncoder(nn.Module):
    """Uses the same parameter namespaces, without duplicating a large model."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        self.classifier = nn.Linear(4, 2)


class KimiaNetWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "weights.pth"
        with torch.random.fork_rng():
            torch.manual_seed(5)
            self.model = TinyEncoder().train()
        self.expected = OrderedDict(
            (name, torch.full_like(value, 2))
            for name, value in self.model.state_dict().items()
        )

    @staticmethod
    def official_names(state, data_parallel=True):
        renamed = OrderedDict()
        for name, value in state.items():
            if name.startswith("features."):
                name = "model." + name[len("features."):]
            elif name.startswith("classifier."):
                name = "fc_4." + name[len("classifier."):]
            renamed[("module." if data_parallel else "") + name] = value
        return renamed

    def test_official_and_native_key_names_load_exactly_and_freeze_model(self):
        for checkpoint in (self.expected, self.official_names(self.expected),
                           self.official_names(self.expected, data_parallel=False)):
            with self.subTest(first_key=next(iter(checkpoint))):
                self.model.train().requires_grad_(True)
                torch.save(checkpoint, self.path)
                loaded = load_kimianet_weights(self.model, self.path)
                self.assertIs(loaded, self.model)
                self.assertFalse(loaded.training)
                self.assertTrue(all(not module.training for module in loaded.modules()))
                self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
                for name, actual in loaded.state_dict().items():
                    self.assertTrue(torch.equal(actual, self.expected[name]), name)

    def test_torch_load_is_explicitly_restricted_to_weights_only_on_cpu(self):
        torch.save(self.expected, self.path)
        original_load = torch.load
        with patch("lung_wsi_mil.features.kimianet.torch.load", wraps=original_load) as load:
            load_kimianet_weights(self.model, self.path)
        load.assert_called_once_with(self.path, map_location="cpu", weights_only=True)

    def test_optional_checksum_verifies_exact_checkpoint_bytes(self):
        torch.save(self.expected, self.path)
        checksum = hashlib.sha256(self.path.read_bytes()).hexdigest()
        load_kimianet_weights(self.model, self.path, expected_sha256=checksum.upper())
        with patch("lung_wsi_mil.features.kimianet.torch.load") as load:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_kimianet_weights(self.model, self.path, expected_sha256="0" * 64)
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "hexadecimal digest"):
            load_kimianet_weights(self.model, self.path, expected_sha256="not a digest")

    def test_missing_checkpoint_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_kimianet_weights(self.model, self.path)

    def test_malformed_mappings_are_rejected(self):
        for checkpoint in ([torch.ones(1)], {"state_dict": self.expected}, {1: torch.ones(1)}):
            with self.subTest(kind=type(checkpoint).__name__):
                torch.save(checkpoint, self.path)
                with self.assertRaisesRegex(ValueError, "flat mapping"):
                    load_kimianet_weights(self.model, self.path)

    def test_converted_key_collisions_are_rejected(self):
        checkpoint = self.official_names(self.expected)
        checkpoint["features.0.weight"] = self.expected["features.0.weight"]
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "keys collide"):
            load_kimianet_weights(self.model, self.path)

    def test_key_shape_and_finiteness_failures_do_not_partly_modify_model(self):
        missing = OrderedDict(self.expected)
        missing.pop("classifier.bias")
        extra = OrderedDict(self.expected, unrelated=torch.ones(1))
        shape = OrderedDict(self.expected)
        shape["classifier.bias"] = torch.zeros(3)
        nonfinite = OrderedDict(self.expected)
        nonfinite["classifier.bias"] = torch.tensor([float("nan"), 1])
        for checkpoint, message in ((missing, "keys do not match"), (extra, "keys do not match"),
                                    (shape, "shape mismatch"), (nonfinite, "non-finite")):
            with self.subTest(message=message):
                before = {name: value.clone() for name, value in self.model.state_dict().items()}
                torch.save(checkpoint, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    load_kimianet_weights(self.model, self.path)
                for name, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
