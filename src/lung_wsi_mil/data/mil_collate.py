"""Mil collate utilities."""

import torch
from torch.nn.utils.rnn import pad_sequence


def mil_collate_fn(batch):
    bags = []
    coordinates = []
    labels = []
    slide_ids = []
    case_ids = []
    is_augmented = []

    for embeddings, coords, label, slide_id, case_id in batch:
        for view_index in range(embeddings.shape[0]):
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
