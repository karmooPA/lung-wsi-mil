import pandas as pd

from src.data.split_storage import save_all_splits


METADATA_PATH = "src/data/dataset_inventory.csv"
OUTPUT_DIR = "splits"


def main():
    metadata = pd.read_csv(METADATA_PATH)

    save_all_splits(
        metadata=metadata,
        output_dir=OUTPUT_DIR,
    )

    print("All splits saved successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()