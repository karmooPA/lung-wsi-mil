"""Mil hdf5 dataset utilities."""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class MILHDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as file:
            self.slide_ids = sorted(file.keys())

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]

        with h5py.File(self.hdf5_path, "r") as file:
            group = file[slide_id]

            required_keys = {
                "embeddings",
                "coords",
                "label",
                "case_id",
            }

            missing_keys = required_keys.difference(group.keys())

            if missing_keys:
                raise KeyError(f"{slide_id}: missing HDF5 datasets: {sorted(missing_keys)}")

            embeddings = torch.tensor(
                group["embeddings"][:],
                dtype=torch.float32,
            )

            coordinates = torch.tensor(
                group["coords"][:],
                dtype=torch.int64,
            )

            label = torch.tensor(
                group["label"][()],
                dtype=torch.float32,
            )

            case_id = group["case_id"][()].decode()

        if embeddings.ndim != 3:
            raise ValueError(
                f"{slide_id}: embeddings must be 3D, got shape {tuple(embeddings.shape)}"
            )

        if embeddings.shape[0] != 3:
            raise ValueError(f"{slide_id}: expected 3 embedding views, got {embeddings.shape[0]}")

        if embeddings.shape[2] != 1024:
            raise ValueError(
                f"{slide_id}: expected embedding dimension 1024, got {embeddings.shape[2]}"
            )

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"{slide_id}: expected coords shape [N, 2], got {tuple(coordinates.shape)}"
            )

        if embeddings.shape[1] != coordinates.shape[0]:
            raise ValueError(
                f"{slide_id}: embedding/coordinate count mismatch: "
                f"{embeddings.shape[1]} embeddings vs "
                f"{coordinates.shape[0]} coordinates"
            )

        if label.ndim != 0:
            raise ValueError(f"{slide_id}: label must be a scalar, got shape {tuple(label.shape)}")

        if label.item() not in (0.0, 1.0):
            raise ValueError(f"{slide_id}: label must be 0 or 1, got {label.item()}")

        return (
            embeddings,
            coordinates,
            label,
            slide_id,
            case_id,
        )
