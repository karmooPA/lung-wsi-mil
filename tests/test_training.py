import pytest
import torch
from torch import nn
from torch.optim import SGD

from lung_wsi_mil.mil.training import train_batch, train_one_epoch


class TinyAMIL(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Linear(input_dim, 1)
        self.forward_calls = 0
        self.bag_sizes = []

    def forward(self, bag):
        self.forward_calls += 1
        self.bag_sizes.append(bag.shape[0])
        pooled = bag.mean(dim=0)
        logit = self.classifier(pooled).squeeze(-1)
        probability = torch.sigmoid(logit)
        prediction = (probability >= 0.5).to(torch.int64)
        attention = torch.full(
            (bag.shape[0],),
            1.0 / bag.shape[0],
            dtype=bag.dtype,
            device=bag.device,
        )
        return probability, prediction, attention


def make_batch(
    bags,
    labels,
    slide_ids=None,
    case_ids=None,
    augmented=None,
):
    batch_size = len(bags)
    max_instances = max(bag.shape[0] for bag in bags)
    feature_dim = bags[0].shape[1]

    padded_bags = torch.full(
        (batch_size, max_instances, feature_dim),
        -torch.inf,
        dtype=torch.float32,
    )
    coordinates = torch.full(
        (batch_size, max_instances, 2),
        -1,
        dtype=torch.long,
    )

    for index, bag in enumerate(bags):
        padded_bags[index, : bag.shape[0]] = bag
        coordinates[index, : bag.shape[0]] = torch.arange(bag.shape[0]).unsqueeze(1).expand(-1, 2)

    if slide_ids is None:
        slide_ids = tuple(f"slide-{index}" for index in range(batch_size))

    if case_ids is None:
        case_ids = tuple(f"case-{index}" for index in range(batch_size))

    if augmented is None:
        augmented = [False] * batch_size

    return (
        padded_bags,
        coordinates,
        torch.tensor(labels, dtype=torch.float32),
        slide_ids,
        case_ids,
        torch.tensor(augmented, dtype=torch.bool),
    )


def make_model_and_optimizer():
    torch.manual_seed(7)
    model = TinyAMIL()
    optimizer = SGD(model.parameters(), lr=0.1)
    return model, optimizer


def make_valid_batch():
    return make_batch(
        bags=[
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
        ],
        labels=[1, 0],
    )


def test_train_batch_returns_expected_metrics():
    model, optimizer = make_model_and_optimizer()

    result = train_batch(model, optimizer, make_valid_batch())

    assert set(result) == {"loss", "num_bags", "num_slides"}
    assert result["num_bags"] == 2
    assert result["num_slides"] == 2
    assert result["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_train_batch_updates_model_parameters():
    model, optimizer = make_model_and_optimizer()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    train_batch(model, optimizer, make_valid_batch())

    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters()))


def test_train_batch_removes_padding_before_forward():
    model, optimizer = make_model_and_optimizer()
    batch = make_batch(
        bags=[
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ],
        labels=[1, 0],
    )

    train_batch(model, optimizer, batch)

    assert model.bag_sizes == [2, 1]


def test_train_batch_processes_each_view_as_a_separate_bag():
    model, optimizer = make_model_and_optimizer()
    bag = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    batch = make_batch(
        bags=[bag, bag.clone(), bag.clone()],
        labels=[1, 1, 1],
        slide_ids=("slide-1", "slide-1", "slide-1"),
        case_ids=("case-1", "case-1", "case-1"),
        augmented=[False, True, True],
    )

    result = train_batch(model, optimizer, batch)

    assert model.forward_calls == 3
    assert result["num_bags"] == 3
    assert result["num_slides"] == 1


def test_train_batch_sets_model_to_training_mode():
    model, optimizer = make_model_and_optimizer()
    model.eval()

    train_batch(model, optimizer, make_valid_batch())

    assert model.training


def test_train_batch_rejects_invalid_batch_length():
    model, optimizer = make_model_and_optimizer()

    with pytest.raises(ValueError, match="six outputs"):
        train_batch(model, optimizer, (torch.ones(1),))


def test_train_batch_rejects_non_tensor_bags():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[0] = [[1.0, 2.0]]

    with pytest.raises(TypeError, match="padded_bags"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_invalid_bag_dimensions():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[0] = torch.ones(2, 2)

    with pytest.raises(ValueError, match=r"shape \[B, N, D\]"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_feature_dimension_mismatch():
    model, optimizer = make_model_and_optimizer()
    batch = make_batch(
        bags=[torch.ones(2, 3)],
        labels=[1],
    )

    with pytest.raises(ValueError, match="feature dimension"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_coordinate_shape_mismatch():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[1] = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="coordinates"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_label_count_mismatch():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[2] = torch.tensor([1.0])

    with pytest.raises(ValueError, match="Expected 2 labels"):
        train_batch(model, optimizer, batch)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_train_batch_rejects_invalid_labels(labels):
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[2] = labels

    with pytest.raises(ValueError):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_wrong_slide_id_count():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[3] = ("slide-1",)

    with pytest.raises(ValueError, match="slide_ids and case_ids"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_non_string_case_ids():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[4] = ("case-1", 2)

    with pytest.raises(TypeError, match="case_id"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_non_boolean_augmentation_flags():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[5] = torch.tensor([0, 1])

    with pytest.raises(TypeError, match="torch.bool"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_wrong_augmentation_shape():
    model, optimizer = make_model_and_optimizer()
    batch = list(make_valid_batch())
    batch[5] = torch.tensor([False])

    with pytest.raises(ValueError, match="shape"):
        train_batch(model, optimizer, batch)


def test_train_batch_rejects_invalid_optimizer():
    model, _ = make_model_and_optimizer()

    with pytest.raises(TypeError, match="torch.optim.Optimizer"):
        train_batch(model, object(), make_valid_batch())


def test_train_batch_rejects_model_without_parameters():
    class ParameterlessModel(nn.Module):
        def forward(self, bag):
            return bag.mean(), None, None

    model = ParameterlessModel()
    optimizer = SGD([nn.Parameter(torch.tensor(1.0))], lr=0.1)

    with pytest.raises(ValueError, match="must contain parameters"):
        train_batch(model, optimizer, make_valid_batch())


def test_train_batch_rejects_invalid_model_output():
    class InvalidOutputModel(TinyAMIL):
        def forward(self, bag):
            return torch.tensor(0.5), torch.tensor(1)

    model = InvalidOutputModel()
    optimizer = SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="AMIL must return"):
        train_batch(model, optimizer, make_valid_batch())


def test_train_batch_rejects_multiple_probabilities_per_bag():
    class InvalidProbabilityModel(TinyAMIL):
        def forward(self, bag):
            return torch.tensor([0.4, 0.6]), None, None

    model = InvalidProbabilityModel()
    optimizer = SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="exactly one probability"):
        train_batch(model, optimizer, make_valid_batch())


def test_train_one_epoch_returns_expected_metrics():
    model, optimizer = make_model_and_optimizer()

    first_batch = make_batch(
        bags=[torch.tensor([[1.0, 2.0]])],
        labels=[1],
    )
    second_batch = make_batch(
        bags=[
            torch.tensor([[-1.0, -2.0]]),
            torch.tensor([[2.0, 1.0]]),
        ],
        labels=[0, 1],
    )

    result = train_one_epoch(
        model,
        [first_batch, second_batch],
        optimizer,
    )

    assert set(result) == {
        "loss",
        "num_batches",
        "num_bags",
        "num_slides",
    }
    assert result["num_batches"] == 2
    assert result["num_bags"] == 3
    assert result["num_slides"] == 3
    assert result["loss"] >= 0.0
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_train_one_epoch_rejects_empty_dataloader():
    model, optimizer = make_model_and_optimizer()

    with pytest.raises(ValueError, match="produced no bags"):
        train_one_epoch(model, [], optimizer)


def test_train_one_epoch_rejects_non_iterable_dataloader():
    model, optimizer = make_model_and_optimizer()

    with pytest.raises(TypeError, match="must be iterable"):
        train_one_epoch(model, 42, optimizer)


def test_train_one_epoch_rejects_invalid_optimizer():
    model, _ = make_model_and_optimizer()

    with pytest.raises(TypeError, match="torch.optim.Optimizer"):
        train_one_epoch(model, [], object())
