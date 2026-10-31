"""Splits utilities."""

from sklearn.model_selection import StratifiedGroupKFold

REFERENCE_SEEDS = [2, 3, 5, 7, 11]


def validate_patient_split(
    metadata,
    train_df,
    test_df,
):
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    patient_overlap = train_patients & test_patients

    if patient_overlap:
        raise ValueError(
            "Patient leakage detected. "
            f"Patients present in both train and test: "
            f"{sorted(patient_overlap)}"
        )

    train_slides = set(train_df["slide_id"])
    test_slides = set(test_df["slide_id"])

    slide_overlap = train_slides & test_slides

    if slide_overlap:
        raise ValueError(
            "Slide leakage detected. "
            f"Slides present in both train and test: "
            f"{sorted(slide_overlap)}"
        )

    assigned_slides = train_slides | test_slides
    expected_slides = set(metadata["slide_id"])

    if assigned_slides != expected_slides:
        missing_slides = expected_slides - assigned_slides
        unexpected_slides = assigned_slides - expected_slides

        raise ValueError(
            "Split does not cover the dataset correctly. "
            f"Missing slides: {sorted(missing_slides)}; "
            f"Unexpected slides: {sorted(unexpected_slides)}"
        )


def create_patient_aware_split(
    metadata,
    seed,
):
    required_columns = {
        "slide_id",
        "patient_id",
        "label",
    }

    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    train_indices, test_indices = next(
        splitter.split(
            X=metadata,
            y=metadata["label"],
            groups=metadata["patient_id"],
        )
    )

    train_df = metadata.iloc[train_indices].copy()
    test_df = metadata.iloc[test_indices].copy()

    validate_patient_split(
        metadata,
        train_df,
        test_df,
    )

    return train_df, test_df


def create_all_reference_splits(metadata):
    splits = {}

    for seed in REFERENCE_SEEDS:
        train_df, test_df = create_patient_aware_split(
            metadata=metadata,
            seed=seed,
        )

        splits[seed] = {
            "train": train_df,
            "test": test_df,
        }

    return splits
