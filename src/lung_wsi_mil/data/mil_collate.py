"""Mil collate utilities."""

import torch
from torch.nn.utils.rnn import pad_sequence


def mil_collate_fn(batch, *, view_indices=None):
    """Expand selected views; the default preserves Original/Aug1/Aug2.

    Validation and test loaders explicitly pass ``view_indices=(0,)`` so one
    slide contributes one prediction without augmented test-time views.
    """
    if not batch:
        raise ValueError("Cannot collate an empty MIL batch.")
    indices = (0, 1, 2) if view_indices is None else tuple(view_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or any(index not in (0, 1, 2) for index in indices)
    ):
        raise ValueError("view_indices must contain distinct indices from (0, 1, 2).")
    bags = []
    coordinates = []
    labels = []
    slide_ids = []
    case_ids = []
    is_augmented = []

    for embeddings, coords, label, slide_id, case_id in batch:
        if embeddings.shape[0] != 3:
            raise ValueError("MIL samples must contain Original, Aug1, and Aug2.")
        for view_index in indices:
            bags.append(embeddings[view_index])
            coordinates.append(coords)
            labels.append(label)
            slide_ids.append(slide_id)
            case_ids.append(case_id)

            is_augmented.append(view_index != 0)

    padded_bags = pad_sequence(
        bags,
        batch_first=True,
        padding_value=float("-inf"),
    )

    padded_coordinates = pad_sequence(
        coordinates,
        batch_first=True,
        padding_value=-1,
    )

    labels = torch.stack(labels)

    is_augmented = torch.tensor(
        is_augmented,
        dtype=torch.bool,
    )

    return (
        padded_bags,
        padded_coordinates,
        labels,
        slide_ids,
        case_ids,
        is_augmented,
    )
