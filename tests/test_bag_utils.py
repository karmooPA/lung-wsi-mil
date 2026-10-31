
import pytest
import torch

from lung_wsi_mil.mil.bag_utils import remove_padding


def test_remove_padding_removes_complete_negative_infinity_rows():
    bag = torch.tensor(
        [[1.0, 2.0], [-torch.inf, -torch.inf], [3.0, 4.0]],
        dtype=torch.float32,
    )

    result = remove_padding(bag)

    expected = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )

    assert torch.equal(result, expected)


def test_remove_padding_preserves_order_dtype_and_device():
    bag = torch.tensor(
        [[4.0, 5.0], [-torch.inf, -torch.inf], [1.0, 2.0]],
        dtype=torch.float64,
    )

    result = remove_padding(bag)

    assert result.dtype == torch.float64
    assert result.device == bag.device
    assert torch.equal(
        result,
        torch.tensor([[4.0, 5.0], [1.0, 2.0]], dtype=torch.float64),
    )


def test_remove_padding_does_not_modify_input():
    bag = torch.tensor(
        [[1.0, 2.0], [-torch.inf, -torch.inf]],
    )
    original = bag.clone()

    remove_padding(bag)

    assert torch.equal(bag, original)


def test_remove_padding_preserves_gradient_flow():
    bag = torch.tensor(
        [[1.0, 2.0], [-torch.inf, -torch.inf], [3.0, 4.0]],
        requires_grad=True,
    )

    remove_padding(bag).sum().backward()

    expected_grad = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
    )

    assert torch.equal(bag.grad, expected_grad)


def test_remove_padding_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="torch.Tensor"):
        remove_padding([[1.0, 2.0]])


def test_remove_padding_rejects_non_floating_tensor():
    with pytest.raises(TypeError, match="floating-point"):
        remove_padding(torch.tensor([[1, 2]], dtype=torch.int64))


@pytest.mark.parametrize(
    "bag",
    [
        torch.tensor([1.0, 2.0]),
        torch.ones((1, 2, 3)),
    ],
)
def test_remove_padding_rejects_invalid_rank(bag):
    with pytest.raises(ValueError, match=r"shape \[N, feature_dim\]"):
        remove_padding(bag)


def test_remove_padding_rejects_zero_instance_rows():
    with pytest.raises(ValueError, match="at least one instance"):
        remove_padding(torch.empty((0, 2), dtype=torch.float32))


def test_remove_padding_rejects_zero_features():
    with pytest.raises(ValueError, match="at least one feature"):
        remove_padding(torch.empty((2, 0), dtype=torch.float32))


def test_remove_padding_rejects_all_padding_bag():
    bag = torch.full((3, 2), -torch.inf)

    with pytest.raises(ValueError, match="no valid instances"):
        remove_padding(bag)


@pytest.mark.parametrize(
    "invalid_row",
    [
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [-float("inf"), 1.0],
        [1.0, -float("inf")],
    ],
)
def test_remove_padding_rejects_invalid_values_in_non_padding_rows(
    invalid_row,
):
    bag = torch.tensor(
        [[1.0, 2.0], invalid_row],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="only finite values"):
        remove_padding(bag)
