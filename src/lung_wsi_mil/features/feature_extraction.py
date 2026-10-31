"""Feature extraction utilities."""

import torch


def extract_embeddings(
    transformed_patches,
    model,
    device,
    batch_size=64,
):
    if transformed_patches.ndim != 5:
        raise ValueError("Expected transformed_patches with shape [3, N, 3, 512, 512].")

    if transformed_patches.shape[0] != 3:
        raise ValueError("Expected 3 representations: Original, Aug1, Aug2.")

    number_of_versions = transformed_patches.shape[0]
    number_of_patches = transformed_patches.shape[1]

    if number_of_patches == 0:
        return torch.empty(
            (number_of_versions, 0, model.feature_dim),
            dtype=torch.float32,
        )

    all_versions_embeddings = []

    for version_index in range(number_of_versions):
        version_patches = transformed_patches[version_index]

        version_embeddings = []

        for start_index in range(
            0,
            number_of_patches,
            batch_size,
        ):
            end_index = start_index + batch_size

            batch = version_patches[start_index:end_index].to(device)

            with torch.inference_mode():
                embeddings, _ = model(batch)

            version_embeddings.append(embeddings.cpu())

        version_embeddings = torch.cat(
            version_embeddings,
            dim=0,
        )

        all_versions_embeddings.append(version_embeddings)

    return torch.stack(
        all_versions_embeddings,
        dim=0,
    )


def extract_embeddings_with_coordinates(
    transformed_patches,
    coordinates,
    model,
    device,
    batch_size=64,
):
    number_of_patches = transformed_patches.shape[1]

    if len(coordinates) != number_of_patches:
        raise ValueError("Number of coordinates must match number of patches.")

    embeddings = extract_embeddings(
        transformed_patches=transformed_patches,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    coordinates = torch.as_tensor(
        coordinates,
        dtype=torch.int64,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape [N, 2].")

    return embeddings, coordinates
