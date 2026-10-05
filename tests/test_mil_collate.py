import torch

from src.data.mil_collate import mil_collate_fn


def test_mil_collate_fn():
    # WSI 1: 5 patches
    embeddings_1 = torch.randn(3, 5, 1024)
    coords_1 = torch.tensor(
        [
            [0, 0],
            [512, 0],
            [1024, 0],
            [0, 512],
            [512, 512],
        ],
        dtype=torch.int64,
    )
    label_1 = torch.tensor(0.0)

    # WSI 2: 8 patches
    embeddings_2 = torch.randn(3, 8, 1024)
    coords_2 = torch.tensor(
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
    )
    label_2 = torch.tensor(1.0)

    batch = [
        (
            embeddings_1,
            coords_1,
            label_1,
            "slide_1",
            "case_1",
        ),
        (
            embeddings_2,
            coords_2,
            label_2,
            "slide_2",
            "case_2",
        ),
    ]

    (
        padded_bags,
        padded_coords,
        labels,
        slide_ids,
        case_ids,
        is_augmented,
    ) = mil_collate_fn(batch)

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
    