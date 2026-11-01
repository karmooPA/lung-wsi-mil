
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.mil_collate import mil_collate_fn


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_split_frame(split_df, name):
    if not hasattr(split_df, "columns") or not hasattr(split_df, "__len__"):
        raise TypeError(f"{name} must be a pandas-like DataFrame.")

    required_columns = {"slide_id", "patient_id", "label"}
    missing_columns = required_columns.difference(split_df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing_columns)}."
        )

    if len(split_df) == 0:
        raise ValueError(f"{name} must not be empty.")

    if split_df[list(required_columns)].isna().any().any():
        raise ValueError(f"{name} contains missing values.")

    if split_df["slide_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate slide IDs.")

    if not split_df["label"].isin([0, 1]).all():
        raise ValueError(f"{name} labels must contain only 0 and 1.")


def _validate_disjoint_splits(first_df, second_df, first_name, second_name):
    first_slides = set(first_df["slide_id"])
    second_slides = set(second_df["slide_id"])

    slide_overlap = first_slides.intersection(second_slides)

    if slide_overlap:
        raise ValueError(
            f"Slide leakage detected between {first_name} and {second_name}: "
            f"{len(slide_overlap)} overlapping slides."
        )

    first_patients = set(first_df["patient_id"])
    second_patients = set(second_df["patient_id"])

    patient_overlap = first_patients.intersection(second_patients)

    if patient_overlap:
        raise ValueError(
            f"Patient leakage detected between {first_name} and {second_name}: "
            f"{len(patient_overlap)} overlapping patients."
        )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataset_subset(dataset, split_df):
    if not hasattr(dataset, "slide_ids"):
        raise TypeError("dataset must expose slide_ids.")

    _validate_split_frame(split_df, "split_df")

    dataset_slide_ids = list(dataset.slide_ids)

    if len(set(dataset_slide_ids)) != len(dataset_slide_ids):
        raise ValueError("Dataset contains duplicate slide IDs.")

    slide_to_index = {
        slide_id: index
        for index, slide_id in enumerate(dataset_slide_ids)
    }

    requested_slide_ids = split_df["slide_id"].tolist()

    missing_slide_ids = [
        slide_id
        for slide_id in requested_slide_ids
        if slide_id not in slide_to_index
    ]

    if missing_slide_ids:
        raise ValueError(
            "Some split slides are missing from the HDF5 dataset: "
            f"{missing_slide_ids}"
        )

    subset_indices = [
        slide_to_index[slide_id]
        for slide_id in requested_slide_ids
    ]

    return Subset(dataset, subset_indices)


def create_mil_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=0,
    generator=None,
):
    _validate_positive_integer(batch_size, "batch_size")
    _validate_non_negative_integer(num_workers, "num_workers")

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")

    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mil_collate_fn,
        generator=generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )


def create_train_val_dataloaders(
    dataset,
    fold_train_df,
    fold_val_df,
    batch_size,
    num_workers=0,
    seed=0,
):
    _validate_positive_integer(batch_size, "batch_size")
    _validate_non_negative_integer(num_workers, "num_workers")
    _validate_non_negative_integer(seed, "seed")

    _validate_split_frame(fold_train_df, "fold_train_df")
    _validate_split_frame(fold_val_df, "fold_val_df")

    _validate_disjoint_splits(
        fold_train_df,
        fold_val_df,
        "training",
        "validation",
    )

    train_dataset = create_dataset_subset(
        dataset,
        fold_train_df,
    )

    val_dataset = create_dataset_subset(
        dataset,
        fold_val_df,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = create_mil_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )

    val_loader = create_mil_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def create_fold_dataloaders(
    dataset,
    fold_train_df,
    fold_val_df,
    test_df,
    batch_size,
    num_workers=0,
    seed=0,
):
    _validate_split_frame(test_df, "test_df")

    _validate_disjoint_splits(
        fold_train_df,
        test_df,
        "training",
        "test",
    )

    _validate_disjoint_splits(
        fold_val_df,
        test_df,
        "validation",
        "test",
    )

    train_loader, val_loader = create_train_val_dataloaders(
        dataset=dataset,
        fold_train_df=fold_train_df,
        fold_val_df=fold_val_df,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    test_dataset = create_dataset_subset(
        dataset,
        test_df,
    )

    test_loader = create_mil_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
