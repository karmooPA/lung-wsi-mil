import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.mil_dataloader import create_fold_dataloaders


class DummyMILDataset(Dataset):
    def __init__(self):
        self.slide_ids = [
            "slide_1",
            "slide_2",
            "slide_3",
            "slide_4",
            "slide_5",
            "slide_6",
        ]

        self.samples = [
            self._make_sample("slide_1", "case_1", 0.0, 5),
            self._make_sample("slide_2", "case_2", 1.0, 8),
            self._make_sample("slide_3", "case_3", 0.0, 6),
            self._make_sample("slide_4", "case_4", 1.0, 7),
            self._make_sample("slide_5", "case_5", 0.0, 4),
            self._make_sample("slide_6", "case_6", 1.0, 9),
        ]

    def _make_sample(
        self,
        slide_id,
        case_id,
        label,
        num_patches,
    ):
        embeddings = torch.randn(
            3,
            num_patches,
            1024,
        )

        coords = torch.arange(
            num_patches * 2,
            dtype=torch.int64,
        ).reshape(num_patches, 2)

        label = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return (
            embeddings,
            coords,
            label,
            slide_id,
            case_id,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_create_fold_dataloaders():
    dataset = DummyMILDataset()

    fold_train_df = pd.DataFrame(
        {
            "slide_id": [
                "slide_1",
                "slide_2",
                "slide_3",
            ]
        }
    )

    fold_val_df = pd.DataFrame(
        {
            "slide_id": [
                "slide_4",
            ]
        }
    )

    test_df = pd.DataFrame(
        {
            "slide_id": [
                "slide_5",
                "slide_6",
            ]
        }
    )

    train_loader, val_loader, test_loader = (
        create_fold_dataloaders(
            dataset=dataset,
            fold_train_df=fold_train_df,
            fold_val_df=fold_val_df,
            test_df=test_df,
            batch_size=2,
            num_workers=0,
        )
    )

    assert len(train_loader.dataset) == 3
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 2

    train_batch = next(iter(train_loader))

    (
        train_bags,
        train_coords,
        train_labels,
        train_slide_ids,
        train_case_ids,
        train_is_augmented,
    ) = train_batch

    assert train_bags.shape[0] == 6
    assert train_bags.shape[2] == 1024

    assert train_coords.shape[0] == 6
    assert train_coords.shape[2] == 2

    assert train_labels.shape == (6,)
    assert len(train_slide_ids) == 6
    assert len(train_case_ids) == 6
    assert train_is_augmented.shape == (6,)

    val_batch = next(iter(val_loader))

    assert val_batch[0].shape[0] == 3
    assert val_batch[2].shape == (3,)

    test_batch = next(iter(test_loader))

    assert test_batch[0].shape[0] == 6
    assert test_batch[2].shape == (6,)