"""Small synthetic mixed-label patients; never a substitute for TCGA evidence."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from lung_wsi_mil.data.split_storage import save_all_splits


def make_synthetic_inputs(root, *, patients=10):
    """Create explicitly synthetic features and geometry in a new/empty folder."""
    root = Path(root)
    if isinstance(patients, bool) or not isinstance(patients, int) or patients < 10:
        raise ValueError(
            "The five-fold software smoke cohort needs at least ten mixed-label patients."
        )
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Synthetic inputs already exist: {root}")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    features = root / "synthetic_features.h5"
    patches = root / "synthetic_patches"
    with h5py.File(features, "w") as file:
        for patient in range(patients):
            for label in (0, 1):
                slide = f"slide-{patient:03d}-{label}"
                count = 3 + (patient + label) % 2
                embeddings = rng.normal(0, 0.1, size=(3, count, 1024)).astype(np.float32)
                embeddings[:, :, :8] += (2 * label - 1) * 0.4
                coordinates = np.asarray(
                    [[64 * index, 0] for index in range(count)], dtype=np.int64
                )
                group = file.create_group(slide)
                group.create_dataset("embeddings", data=embeddings)
                group.create_dataset("coords", data=coordinates)
                group.create_dataset("label", data=np.int64(label))
                group.create_dataset("case_id", data=f"patient-{patient:03d}".encode())
                rows.append(
                    {
                        "slide_id": slide,
                        "patient_id": f"patient-{patient:03d}",
                        "label": "Tumor" if label else "Normal",
                    }
                )
                pre = patches / "slides" / slide / "preprocessing"
                (pre / "qc").mkdir(parents=True)
                np.save(pre / "coordinates.npy", coordinates, allow_pickle=False)
                image = Image.new("RGB", (256, 128), "#F0F1F4")
                draw = ImageDraw.Draw(image)
                for index in range(count):
                    draw.rectangle(
                        (64 * index, 0, 64 * index + 63, 63),
                        fill=(178 + 10 * index, 142 + 5 * index, 182 + 6 * index),
                    )
                draw.text((12, 94), "SYNTHETIC GEOMETRY", fill="#172B3D")
                image.save(pre / "qc/thumbnail.png")
                (pre / "result.json").write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "result": {
                                "slide_info": {"dimensions": [256, 128]},
                                "resolution": {"level0_patch_size": 64.0},
                                "valid_patch_count": count,
                            },
                            "files": {"qc": {"thumbnail": "qc/thumbnail.png"}},
                        }
                    )
                    + "\n"
                )
    inventory = root / "inventory.csv"
    metadata = pd.DataFrame(rows)
    metadata.to_csv(inventory, index=False)
    splits = root / "splits"
    save_all_splits(metadata, splits)
    return {
        "features": features,
        "inventory": inventory,
        "splits": splits,
        "patch_root": patches,
        "metadata": metadata,
    }
