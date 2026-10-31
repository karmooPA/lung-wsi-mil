"""Pipeline utilities."""

from lung_wsi_mil.features.feature_extraction import (
    extract_embeddings_with_coordinates,
)
from lung_wsi_mil.features.hdf5_writer import save_slide_to_hdf5


def process_transformed_slide(
    transformed_patches,
    coordinates,
    model,
    device,
    hdf5_path,
    slide_id,
    case_id,
    label,
    batch_size=64,
):
    embeddings, coordinates = extract_embeddings_with_coordinates(
        transformed_patches=transformed_patches,
        coordinates=coordinates,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    saved = save_slide_to_hdf5(
        hdf5_path=hdf5_path,
        slide_id=slide_id,
        case_id=case_id,
        label=label,
        embeddings=embeddings,
        coordinates=coordinates,
    )

    return saved
