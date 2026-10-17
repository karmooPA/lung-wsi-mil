from sklearn.model_selection import StratifiedGroupKFold


CV_RANDOM_STATE = 42
N_FOLDS = 5


def validate_fold_split(
    full_train_df,
    fold_train_df,
    fold_val_df,
):
    train_patients = set(
        fold_train_df["patient_id"]
    )

    val_patients = set(
        fold_val_df["patient_id"]
    )

    patient_overlap = train_patients & val_patients

    if patient_overlap:
        raise ValueError(
            "Patient leakage detected inside CV fold. "
            f"Patients in both train and validation: "
            f"{sorted(patient_overlap)}"
        )

    train_slides = set(
        fold_train_df["slide_id"]
    )

    val_slides = set(
        fold_val_df["slide_id"]
    )

    slide_overlap = train_slides & val_slides

    if slide_overlap:
        raise ValueError(
            "Slide leakage detected inside CV fold. "
            f"Slides in both train and validation: "
            f"{sorted(slide_overlap)}"
        )

    assigned_slides = train_slides | val_slides
    expected_slides = set(
        full_train_df["slide_id"]
    )

    if assigned_slides != expected_slides:
        raise ValueError(
            "Fold does not cover the full Train/CV set."
        )


def create_patient_aware_folds(train_df):
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    folds = []

    for fold_index, (train_indices, val_indices) in enumerate(
        splitter.split(
            X=train_df,
            y=train_df["label"],
            groups=train_df["patient_id"],
        )
    ):
        fold_train_df = train_df.iloc[train_indices].copy()
        fold_val_df = train_df.iloc[val_indices].copy()

        validate_fold_split(
            full_train_df=train_df,
            fold_train_df=fold_train_df,
            fold_val_df=fold_val_df,
        )

        folds.append(
            {
                "fold": fold_index + 1,
                "train": fold_train_df,
                "val": fold_val_df,
            }
        )

    return folds