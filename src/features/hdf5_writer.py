from pathlib import Path

import h5py
import numpy as np
import torch


def save_slide_to_hdf5(
    hdf5_path,
    slide_id,
    case_id,
    label,
    embeddings,
    coordinates,
):
    hdf5_path = Path(hdf5_path)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.detach().cpu().numpy()

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    coordinates = np.asarray(
        coordinates,
        dtype=np.int64,
    )

    if embeddings.ndim != 3:
        raise ValueError(
            "Embeddings must have shape [3, N, 1024]."
        )

    if embeddings.shape[0] != 3:
        raise ValueError(
            "Expected 3 embedding versions: Original, Aug1, Aug2."
        )

    if embeddings.shape[2] != 1024:
        raise ValueError(
            "Expected KimiaNet embedding dimension of 1024."
        )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "Coordinates must have shape [N, 2]."
        )

    if embeddings.shape[1] != coordinates.shape[0]:
        raise ValueError(
            "Number of embeddings and coordinates must match."
        )

    hdf5_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(hdf5_path, "a") as file:
        if slide_id in file:
            return False

        group = file.create_group(slide_id)

        group.create_dataset(
            "slide_id",
            data=slide_id,
        )

        group.create_dataset(
            "case_id",
            data=case_id,
        )

        group.create_dataset(
            "coords",
            data=coordinates,
        )

        group.create_dataset(
            "embeddings",
            data=embeddings,
            dtype=np.float32,
            chunks=True,
        )

        group.create_dataset(
            "label",
            data=np.asarray(label),
        )

    return True
