import torch
from torch.utils.data import Dataset

from src.data.mil_dataloader import create_mil_dataloader


class DummyMILDataset(Dataset):
    def __init__(self):
        self.samples = [
            (
                torch.randn(3, 5, 1024),
                torch.tensor(
                    [
                        [0, 0],
                        [512, 0],
                        [1024, 0],
                        [0, 512],
                        [512, 512],
                    ],
                    dtype=torch.int64,
                ),
                torch.tensor(0.0),
                "slide_1",
                "case_1",
            ),
            (
                torch.randn(3, 8, 1024),
                torch.tensor(
                    [
                        [0, 0],
                        [512, 0],
                        [1024, 0],
                        [1536, 0],
                        [0, 512],
                        [512, 512],
                        [1024, 512],
                        [1536, 512],
                    ],
                    dtype=torch.int64,
                ),
                torch.tensor(1.0),
                "slide_2",
                "case_2",
            ),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_create_mil_dataloader():
    dataset = DummyMILDataset()

    loader = create_mil_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    (
        padded_bags,
        padded_coords,
        labels,
        slide_ids,
        case_ids,
        is_augmented,
    ) = batch

    assert padded_bags.shape == (6, 8, 1024)
    assert padded_coords.shape == (6, 8, 2)

    assert labels.shape == (6,)

    assert slide_ids == [
        "slide_1",
        "slide_1",
        "slide_1",
        "slide_2",
        "slide_2",
        "slide_2",
    ]

    assert case_ids == [
        "case_1",
        "case_1",
        "case_1",
        "case_2",
        "case_2",
        "case_2",
    ]

    assert is_augmented.tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]