#!/usr/bin/env python3
"""Save a denormalized Original/Aug1/Aug2 QC panel with input provenance."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.augmentation.data_augmentations import IMAGENET_MEAN, IMAGENET_STD, transform_patch


def make_preview(patch_path, output, seed):
    patch_path, output = Path(patch_path), Path(output)
    if output.resolve() == patch_path.resolve():
        raise ValueError("QC output must not overwrite the input patch")
    with Image.open(patch_path) as image:
        # Deliberately reject wrong sizes/modes; a thumbnail is not a 5x patch.
        views = transform_patch(image, rng=np.random.default_rng(seed))
    panel = Image.new("RGB", (3 * 512, 552), "white")
    draw = ImageDraw.Draw(panel)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    for index, (name, tensor) in enumerate(zip(("Original", "Aug1", "Aug2"), views)):
        if tensor.shape != (3, 512, 512) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Invalid transformed tensor")
        pixels = ((tensor * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)
        panel.paste(Image.fromarray(pixels.permute(1, 2, 0).numpy()), (index * 512, 40))
        draw.text((index * 512 + 12, 12), f"{name} | seed {seed}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    metadata = {
        "input_name": patch_path.name,
        "input_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        "seed": seed,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "Pillow", "scikit-image", "torch", "torchvision")},
        "panel_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "review": "Human review required; input authenticity/magnification is not inferred by this script",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/augmentation/qc.png"))
    args = parser.parse_args()
    make_preview(args.patch, args.output, args.seed)
    print(f"Saved {args.output} and its JSON provenance; review all three views.")


if __name__ == "__main__":
    main()
