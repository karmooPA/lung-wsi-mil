from torch.utils.data import DataLoader

from src.data.mil_collate import mil_collate_fn


def create_mil_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mil_collate_fn,
    )