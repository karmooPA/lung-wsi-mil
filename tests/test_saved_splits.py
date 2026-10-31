from pathlib import Path

import pandas as pd

from lung_wsi_mil.data.splits import (
    REFERENCE_SEEDS,
    validate_patient_split,
)
from lung_wsi_mil.data.cross_validation import (
    N_FOLDS,
    validate_fold_split,
)


PROJECT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT / "metadata" / "dataset_inventory.csv"
SPLITS_DIR = PROJECT / "splits"


def test_saved_splits_are_consistent():
    metadata = pd.read_csv(METADATA_PATH)

    for seed in REFERENCE_SEEDS:
        seed_dir = SPLITS_DIR / f"seed_{seed}"

        train_df = pd.read_csv(
            seed_dir / "train_cv.csv"
        )

        test_df = pd.read_csv(
            seed_dir / "test.csv"
        )

        # Outer split validation
        validate_patient_split(
            metadata=metadata,
            train_df=train_df,
            test_df=test_df,
        )

        validation_slides = []
        validation_patients = []

        for fold_number in range(1, N_FOLDS + 1):
            fold_train_df = pd.read_csv(
                seed_dir / f"fold_{fold_number}_train.csv"
            )

            fold_val_df = pd.read_csv(
                seed_dir / f"fold_{fold_number}_val.csv"
            )

            # Fold validation
            validate_fold_split(
                full_train_df=train_df,
                fold_train_df=fold_train_df,
                fold_val_df=fold_val_df,
            )

            validation_slides.extend(
                fold_val_df["slide_id"].tolist()
            )

            validation_patients.extend(
                fold_val_df["patient_id"].unique().tolist()
            )

        # Every Train/CV slide must appear
        # exactly once as validation.
        assert len(validation_slides) == len(train_df)

        assert len(set(validation_slides)) == len(train_df)

        assert set(validation_slides) == set(
            train_df["slide_id"]
        )

        # Every Train/CV patient must appear
        # exactly once in a validation fold.
        expected_patients = set(
            train_df["patient_id"]
        )

        assert len(validation_patients) == len(
            expected_patients
        )

        assert set(validation_patients) == expected_patients
