import math

import pytest
import torch

from lung_wsi_mil.mil.losses import binary_cross_entropy_loss


def test_binary_cross_entropy_returns_expected_value():
    probabilities = torch.tensor([0.8, 0.2])
    targets = torch.tensor([1, 0])

    loss = binary_cross_entropy_loss(probabilities, targets)

    expected = torch.tensor(-math.log(0.8))

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_binary_cross_entropy_accepts_column_tensors():
    probabilities = torch.tensor([[0.8], [0.2]])
    targets = torch.tensor([[1], [0]])

    loss = binary_cross_entropy_loss(probabilities, targets)

    assert torch.allclose(loss, torch.tensor(-math.log(0.8)), atol=1e-6)


def test_binary_cross_entropy_accepts_single_sample():
    probabilities = torch.tensor(0.8)
    targets = torch.tensor(1)

    loss = binary_cross_entropy_loss(probabilities, targets)

    assert torch.allclose(loss, torch.tensor(-math.log(0.8)), atol=1e-6)


def test_binary_cross_entropy_clamps_boundary_probabilities():
    probabilities = torch.tensor([0.0, 1.0])
    targets = torch.tensor([0, 1])

    loss = binary_cross_entropy_loss(probabilities, targets)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_binary_cross_entropy_supports_backpropagation():
    probabilities = torch.tensor([0.8, 0.2], requires_grad=True)
    targets = torch.tensor([1, 0])

    loss = binary_cross_entropy_loss(probabilities, targets)
    loss.backward()

    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.all(probabilities.grad != 0)


def test_binary_cross_entropy_rejects_non_tensor_probabilities():
    with pytest.raises(TypeError, match="torch.Tensor"):
        binary_cross_entropy_loss([0.8, 0.2], torch.tensor([1, 0]))


def test_binary_cross_entropy_rejects_integer_probabilities():
    with pytest.raises(TypeError, match="floating-point"):
        binary_cross_entropy_loss(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.ones((2, 2)),
        torch.ones((1, 2, 1)),
    ],
)
def test_binary_cross_entropy_rejects_invalid_probability_shapes(
    probabilities,
):
    with pytest.raises(ValueError, match="shape"):
        binary_cross_entropy_loss(
            probabilities,
            torch.ones(probabilities.shape[0]),
        )


def test_binary_cross_entropy_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number"):
        binary_cross_entropy_loss(
            torch.tensor([0.8, 0.2]),
            torch.tensor([1]),
        )


def test_binary_cross_entropy_rejects_empty_probabilities():
    with pytest.raises(ValueError, match="must not be empty"):
        binary_cross_entropy_loss(
            torch.empty(0),
            torch.empty(0),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([-float("inf")]),
    ],
)
def test_binary_cross_entropy_rejects_non_finite_probabilities(
    probabilities,
):
    with pytest.raises(ValueError, match="NaN or Inf"):
        binary_cross_entropy_loss(
            probabilities,
            torch.tensor([1]),
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([-0.1]),
        torch.tensor([1.1]),
    ],
)
def test_binary_cross_entropy_rejects_out_of_range_probabilities(
    probabilities,
):
    with pytest.raises(ValueError, match="between 0 and 1"):
        binary_cross_entropy_loss(
            probabilities,
            torch.tensor([1]),
        )


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([2]),
        torch.tensor([-1]),
        torch.tensor([0.5]),
    ],
)
def test_binary_cross_entropy_rejects_non_binary_targets(targets):
    with pytest.raises(ValueError, match="binary labels"):
        binary_cross_entropy_loss(torch.tensor([0.8]), targets)


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
    ],
)
def test_binary_cross_entropy_rejects_non_finite_targets(targets):
    with pytest.raises(ValueError, match="NaN or Inf"):
        binary_cross_entropy_loss(torch.tensor([0.8]), targets)


def test_binary_cross_entropy_rejects_invalid_target_shape():
    with pytest.raises(ValueError, match="shape"):
        binary_cross_entropy_loss(
            torch.tensor([0.8, 0.2]),
            torch.tensor([[1, 0], [0, 1]]),
        )


@pytest.mark.parametrize(
    "eps",
    [
        0.0,
        0.5,
        -0.1,
        float("inf"),
        float("nan"),
        True,
    ],
)
def test_binary_cross_entropy_rejects_invalid_eps(eps):
    with pytest.raises((TypeError, ValueError)):
        binary_cross_entropy_loss(
            torch.tensor([0.8]),
            torch.tensor([1]),
            eps=eps,
        )


def test_binary_cross_entropy_rejects_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    with pytest.raises(ValueError, match="same device"):
        binary_cross_entropy_loss(
            torch.tensor([0.8], device="cuda"),
            torch.tensor([1], device="cpu"),
        )
