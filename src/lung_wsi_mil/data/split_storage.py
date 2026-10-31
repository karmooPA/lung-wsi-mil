"""Split storage utilities."""

from pathlib import Path

from lung_wsi_mil.data.cross_validation import create_patient_aware_folds
from lung_wsi_mil.data.splits import create_all_reference_splits


def save_all_splits(
    metadata,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_splits = create_all_reference_splits(metadata)

    for seed, split_data in all_splits.items():
        seed_dir = output_dir / f"seed_{seed}"

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df = split_data["train"]
        test_df = split_data["test"]

        train_df.sort_values("slide_id").to_csv(
            seed_dir / "train_cv.csv",
            index=False,
        )

        test_df.sort_values("slide_id").to_csv(
            seed_dir / "test.csv",
            index=False,
        )

        folds = create_patient_aware_folds(train_df)

        for fold_data in folds:
            fold_number = fold_data["fold"]

            fold_train = fold_data["train"]
            fold_val = fold_data["val"]

            fold_train.sort_values("slide_id").to_csv(
                seed_dir / f"fold_{fold_number}_train.csv",
                index=False,
            )

            fold_val.sort_values("slide_id").to_csv(
                seed_dir / f"fold_{fold_number}_val.csv",
                index=False,
            )
