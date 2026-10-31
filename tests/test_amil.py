"""Independent checks for the AMIL model and initialization contract.

These tests intentionally use clean, single bags. Padding and the real HDF5
pipeline are checked separately in the integration suite.
"""

import copy
import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from lung_wsi_mil.mil.amil import AMIL
from lung_wsi_mil.mil.initialization import initialize_weights


class AMILContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.model = AMIL()
        self.model.eval()

    def assert_output_contract(self, output, n, device="cpu"):
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        probability, prediction, attention = output
        for value, shape in zip(output, ((1, 1), (1, 1), (1, n))):
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(tuple(value.shape), shape)
            self.assertEqual(value.device.type, torch.device(device).type)
            self.assertTrue(torch.isfinite(value).all().item())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all().item())
        self.assertTrue(((prediction == 0) | (prediction == 1)).all().item())
        self.assertTrue((attention >= 0).all().item())
        torch.testing.assert_close(
            attention.sum(dim=1),
            torch.ones(1, device=attention.device),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_constructor_defaults_and_single_head_architecture(self):
        defaults = inspect.signature(AMIL).parameters
        self.assertEqual(defaults["input_dim"].default, 1024)
        self.assertEqual(defaults["attention_dim"].default, 128)
        self.assertEqual(defaults["threshold"].default, 0.5)
        self.assertEqual(self.model.input_dim, 1024)
        self.assertEqual(self.model.attention_dim, 128)
        self.assertEqual(self.model.threshold, 0.5)
        linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears], [(1024, 128), (128, 1), (1024, 1)]
        )
        self.assertEqual(sum(isinstance(m, nn.Tanh) for m in self.model.modules()), 1)
        self.assertEqual(sum(isinstance(m, nn.Sigmoid) for m in self.model.modules()), 1)
        self.assertIsInstance(self.model.attention_hidden[0], nn.Linear)
        self.assertIsInstance(self.model.attention_hidden[1], nn.Tanh)
        self.assertIsInstance(self.model.classifier[0], nn.Linear)
        self.assertIsInstance(self.model.classifier[1], nn.Sigmoid)

    def test_constructor_rejects_invalid_dimensions_heads_and_thresholds(self):
        for kwargs in (
            {"input_dim": 0},
            {"input_dim": -1},
            {"attention_dim": 0},
            {"attention_dim": -1},
            {"k": 0},
            {"k": 2},
            {"threshold": -0.01},
            {"threshold": 1.01},
            {"threshold": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AMIL(**kwargs)

    def test_public_tuple_shapes_finite_probability_and_attention(self):
        for n in (1, 7, 31, 200):
            with self.subTest(instances=n):
                self.assert_output_contract(self.model(torch.randn(n, 1024)), n)

    def test_one_instance_receives_attention_one(self):
        _, _, attention = self.model(torch.randn(1, 1024))
        torch.testing.assert_close(attention, torch.ones(1, 1), rtol=0, atol=0)

    def test_attention_matches_independent_linear_tanh_softmax_calculation(self):
        x = torch.randn(13, 1024)
        first = self.model.attention_hidden[0]
        second = self.model.attention_score
        hidden = torch.tanh(F.linear(x, first.weight, first.bias))
        scores = F.linear(hidden, second.weight, second.bias).transpose(0, 1)
        expected = torch.softmax(scores, dim=1)
        _, _, attention = self.model(x)
        torch.testing.assert_close(attention, expected, rtol=1e-6, atol=1e-7)

    def test_weighted_sum_and_classifier_match_independent_calculation(self):
        x = torch.randn(17, 1024)
        probability, _, attention = self.model(x)
        expected_bag = attention @ x
        self.assertEqual(tuple(expected_bag.shape), (1, 1024))
        torch.testing.assert_close(
            probability,
            self.model.classifier(expected_bag),
            rtol=1e-6,
            atol=1e-7,
        )
        classifier = self.model.classifier[0]
        expected_probability = torch.sigmoid(
            F.linear(expected_bag, classifier.weight, classifier.bias)
        )
        torch.testing.assert_close(probability, expected_probability, rtol=1e-6, atol=1e-7)

    def test_threshold_below_equal_above_half_is_inclusive(self):
        x = torch.randn(7, 1024)
        classifier = self.model.classifier[0]
        with torch.no_grad():
            classifier.weight.zero_()
            for logit, expected in ((-1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
                with self.subTest(logit=logit):
                    classifier.bias.fill_(logit)
                    probability, prediction, _ = self.model(x)
                    self.assertEqual(prediction.item(), expected)
                    self.assertEqual(probability.item(), torch.sigmoid(torch.tensor(logit)).item())
                    if logit == 0:
                        self.assertEqual(probability.item(), 0.5)

    def test_prediction_uses_configured_threshold(self):
        model = AMIL(threshold=0.75)
        with torch.no_grad():
            model.classifier[0].weight.zero_()
            model.classifier[0].bias.zero_()
        probability, prediction, _ = model(torch.randn(5, 1024))
        self.assertEqual(probability.item(), 0.5)
        self.assertEqual(prediction.item(), 0.0)

    def test_prediction_is_permutation_invariant_and_attention_is_equivariant(self):
        x = torch.randn(31, 1024)
        permutation = torch.randperm(len(x))
        probability, prediction, attention = self.model(x)
        other_probability, other_prediction, other_attention = self.model(x[permutation])
        torch.testing.assert_close(other_probability, probability, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(other_prediction, prediction, rtol=0, atol=0)
        torch.testing.assert_close(other_attention, attention[:, permutation], rtol=1e-5, atol=1e-6)

    def test_repeated_identical_instances_receive_uniform_attention(self):
        x = torch.randn(1, 1024).repeat(9, 1)
        _, _, attention = self.model(x)
        torch.testing.assert_close(attention, torch.full((1, 9), 1 / 9), rtol=1e-6, atol=1e-7)

    def test_wrong_rank_feature_width_and_empty_bag_raise_clear_errors(self):
        for shape in ((1024,), (2, 3, 1024), (4, 512), (0, 1024)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, r"(?i)(shape|dimension|empty|instance)"),
            ):
                self.model(torch.empty(shape))

    def test_nonfinite_input_and_padding_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            x = torch.randn(7, 1024)
            x[3, 19] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, r"(?i)(finite|nan|inf|padding)"),
            ):
                self.model(x)
        x = torch.cat((torch.randn(3, 1024), torch.full((2, 1024), -torch.inf)))
        with self.assertRaisesRegex(ValueError, r"(?i)(finite|nan|inf|padding)"):
            self.model(x)

    def test_non_tensor_input_is_rejected_explicitly(self):
        for value in (None, [[0.0] * 1024], "not a tensor", 1):
            with (
                self.subTest(type=type(value).__name__),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.model(value)

    def test_nonfloating_tensor_input_is_rejected_explicitly(self):
        for dtype in (torch.int64, torch.int32, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaises((TypeError, ValueError)):
                self.model(torch.ones(3, 1024, dtype=dtype))

    def test_noncontiguous_clean_bag_is_supported(self):
        x = torch.randn(1024, 7).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        self.assert_output_contract(self.model(x), 7)

    def test_all_trainable_parameters_receive_finite_gradients_on_cpu(self):
        self.model.train()
        probability, _, _ = self.model(torch.randn(11, 1024))
        probability.sum().backward()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def _check_accelerator_forward_and_backward(self, device):
        model = AMIL().to(device)
        output = model(torch.randn(11, 1024, device=device))
        self.assert_output_contract(output, 11, device)
        output[0].sum().backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.device.type, device)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_and_backward_when_available(self):
        self._check_accelerator_forward_and_backward("cuda")

    @unittest.skipUnless(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_mps_forward_and_backward_when_available(self):
        self._check_accelerator_forward_and_backward("mps")


class AMILInitializationTests(unittest.TestCase):
    def test_default_initialization_method_is_xavier(self):
        model = nn.Linear(7, 3)
        expected = copy.deepcopy(model)
        torch.manual_seed(111)
        initialize_weights(model)
        torch.manual_seed(111)
        nn.init.xavier_uniform_(expected.weight)
        nn.init.zeros_(expected.bias)
        torch.testing.assert_close(model.weight, expected.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.bias, expected.bias, rtol=0, atol=0)

    def test_both_methods_initialize_all_model_linears_with_zero_bias(self):
        for method in ("xavier", "kaiming"):
            with self.subTest(method=method):
                model = AMIL()
                initialize_weights(model, method=method)
                for layer in model.modules():
                    if isinstance(layer, nn.Linear):
                        self.assertTrue(torch.isfinite(layer.weight).all().item())
                        self.assertGreater(torch.count_nonzero(layer.weight).item(), 0)
                        if layer.bias is not None:
                            torch.testing.assert_close(
                                layer.bias, torch.zeros_like(layer.bias), rtol=0, atol=0
                            )

    def test_methods_match_independent_pytorch_initializers_including_biasless(self):
        template = nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 2, bias=False))
        for method in ("xavier", "kaiming"):
            with self.subTest(method=method):
                actual = copy.deepcopy(template)
                expected = copy.deepcopy(template)
                torch.manual_seed(327)
                initialize_weights(actual, method=method)
                torch.manual_seed(327)
                for layer in expected.modules():
                    if isinstance(layer, nn.Linear):
                        if method == "xavier":
                            nn.init.xavier_uniform_(layer.weight)
                        else:
                            nn.init.kaiming_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                for name, parameter in actual.named_parameters():
                    torch.testing.assert_close(
                        parameter, dict(expected.named_parameters())[name], rtol=0, atol=0
                    )

    def test_initialization_is_reproducible_for_each_method(self):
        for method in ("xavier", "kaiming"):
            with self.subTest(method=method):
                first, second = AMIL(), AMIL()
                torch.manual_seed(409)
                initialize_weights(first, method)
                torch.manual_seed(409)
                initialize_weights(second, method)
                for name, parameter in first.named_parameters():
                    torch.testing.assert_close(
                        parameter, dict(second.named_parameters())[name], rtol=0, atol=0
                    )

    def test_unsupported_method_is_rejected_even_without_linear_layers(self):
        for model in (nn.Linear(7, 3), nn.Identity(), nn.Sequential(nn.Tanh())):
            with (
                self.subTest(model=type(model).__name__),
                self.assertRaisesRegex(ValueError, r"(?i)(method|initializ|unsupported)"),
            ):
                initialize_weights(model, method="unknown")


if __name__ == "__main__":
    unittest.main()
