"""Mil dataloader utilities."""

from torch.utils.data import DataLoader, Subset

from lung_wsi_mil.data.mil_collate import mil_collate_fn


def create_dataset_subset(
    dataset,
    split_df,
):
    slide_to_index = {slide_id: index for index, slide_id in enumerate(dataset.slide_ids)}

    requested_slide_ids = split_df["slide_id"].tolist()

    missing_slide_ids = [
        slide_id for slide_id in requested_slide_ids if slide_id not in slide_to_index
    ]

    if missing_slide_ids:
        raise ValueError(
            f"Some split slides are missing from the HDF5 dataset: {missing_slide_ids}"
        )

    subset_indices = [slide_to_index[slide_id] for slide_id in requested_slide_ids]

    return Subset(
        dataset,
        subset_indices,
    )


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


def create_fold_dataloaders(
    dataset,
    fold_train_df,
    fold_val_df,
    test_df,
    batch_size,
    num_workers=0,
):
    train_dataset = create_dataset_subset(
        dataset,
        fold_train_df,
    )

    val_dataset = create_dataset_subset(
        dataset,
        fold_val_df,
    )

    test_dataset = create_dataset_subset(
        dataset,
        test_df,
    )

    train_loader = create_mil_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_mil_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_mil_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
