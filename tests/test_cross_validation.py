import pandas as pd

from src.data.splits import create_all_reference_splits
from src.data.cross_validation import create_patient_aware_folds


metadata = pd.read_csv(
    "src/data/dataset_inventory.csv"
)

all_splits = create_all_reference_splits(metadata)


for seed, split_data in all_splits.items():
    train_df = split_data["train"]
    test_df = split_data["test"]

    folds = create_patient_aware_folds(train_df)

    validation_slide_ids = []
    validation_patient_ids = []

    print("=" * 70)
    print("Seed:", seed)
    print("Train/CV slides:", len(train_df))
    print("Final Test slides:", len(test_df))

    for fold_data in folds:
        fold_number = fold_data["fold"]
        fold_train = fold_data["train"]
        fold_val = fold_data["val"]

        train_patients = set(fold_train["patient_id"])
        val_patients = set(fold_val["patient_id"])

        patient_overlap = train_patients & val_patients

        validation_slide_ids.extend(
            fold_val["slide_id"].tolist()
        )

        validation_patient_ids.extend(
            fold_val["patient_id"].unique().tolist()
        )

        print()
        print("Fold:", fold_number)

        print(
            "Train slides:",
            len(fold_train)
        )

        print(
            "Validation slides:",
            len(fold_val)
        )

        print(
            "Validation labels:",
            fold_val["label"].value_counts().to_dict()
        )

        print(
            "Patient overlap:",
            len(patient_overlap)
        )

    print()
    print("CV coverage checks")

    print(
        "Validation slides total:",
        len(validation_slide_ids)
    )

    print(
        "Unique validation slides:",
        len(set(validation_slide_ids))
    )

    print(
        "Expected Train/CV slides:",
        len(train_df)
    )

    print(
        "Validation patients total:",
        len(validation_patient_ids)
    )

    print(
        "Unique validation patients:",
        len(set(validation_patient_ids))
    )

    print(
        "Expected Train/CV patients:",
        train_df["patient_id"].nunique()
    )