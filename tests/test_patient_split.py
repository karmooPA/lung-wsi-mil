import pandas as pd

from src.data.splits import create_all_reference_splits


metadata = pd.read_csv(
    "src/data/dataset_inventory.csv"
)

splits = create_all_reference_splits(metadata)


for seed, split in splits.items():
    train_df = split["train"]
    test_df = split["test"]

    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    print("=" * 50)
    print("Seed:", seed)

    print("Total slides:", len(metadata))
    print("Train/CV slides:", len(train_df))
    print("Test slides:", len(test_df))

    print()

    print(
        "Train/CV patients:",
        train_df["patient_id"].nunique()
    )

    print(
        "Test patients:",
        test_df["patient_id"].nunique()
    )

    print()

    print("Train/CV labels:")
    print(train_df["label"].value_counts())

    print()

    print("Test labels:")
    print(test_df["label"].value_counts())

    print()

    print(
        "Train ratio:",
        round(len(train_df) / len(metadata) * 100, 2),
        "%"
    )

    print(
        "Test ratio:",
        round(len(test_df) / len(metadata) * 100, 2),
        "%"
    )

    print(
        "Patient overlap:",
        len(train_patients & test_patients)
    )