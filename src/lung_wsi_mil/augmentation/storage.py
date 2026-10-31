"""Lossless compact storage for three-view augmentation outputs.

The public augmentation API returns normalized float32 tensors with shape
[3, N, 3, 512, 512].  Persisting those tensors directly costs 9 MiB per patch.
Because every view originates from an 8-bit RGB PIL image immediately before
ImageNet normalization, the normalization can be inverted exactly to uint8 and
re-applied later without changing any tensor value.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data_augmentations import IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE

REPRESENTATIONS = ("Original", "Aug1", "Aug2")
SUPPORTED_FORMAT_VERSIONS = {"1.0", "2.0"}
REQUIRED_DATASETS = {"images", "coords", "tissue_percentage"}


def _validate_coordinate_and_tissue_values(coordinates, tissue_values):
    """Reject semantically invalid coordinate/tissue arrays."""
    if bool(np.any(coordinates < 0)):
        raise ValueError("coords contains negative Level-0 coordinates.")
    if not bool(np.isfinite(tissue_values).all()):
        raise ValueError("tissue_percentage contains non-finite values.")
    if bool(np.any((tissue_values < 0) | (tissue_values > 100))):
        raise ValueError("tissue_percentage must be in [0,100].")


def _decode_json_attribute(value, name):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON HDF5 attribute {name!r}.") from error
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"Invalid HDF5 attribute {name!r}.")


def _normalization_tensors(device=None):
    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 1, 1)
    return mean, std


def normalized_views_to_uint8(transformed):
    """Convert [3,N,3,512,512] normalized views to RGB uint8 losslessly.

    Returns a CPU NumPy array with layout [3,N,512,512,3].  Re-applying
    :func:`uint8_views_to_normalized` reconstructs the original float32 tensor
    bit-for-bit for outputs produced by this project's augmentation pipeline.
    """
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("transformed must be a torch.Tensor")
    if transformed.ndim != 5:
        raise ValueError("Expected transformed shape [3, N, 3, 512, 512].")
    if transformed.shape[0] != 3 or transformed.shape[2:] != (3, PATCH_SIZE, PATCH_SIZE):
        raise ValueError("Expected transformed shape [3, N, 3, 512, 512].")
    if transformed.dtype != torch.float32:
        raise ValueError("Expected float32 augmentation output.")
    if not torch.isfinite(transformed).all():
        raise ValueError("Augmentation output contains non-finite values.")

    cpu = transformed.detach().to(device="cpu", dtype=torch.float32)
    mean, std = _normalization_tensors(device=cpu.device)
    pixels = (cpu * std + mean).mul(255.0).round().clamp_(0.0, 255.0)
    return pixels.to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous().numpy()


def uint8_views_to_normalized(images):
    """Restore normalized float32 [3,N,3,512,512] from stored RGB uint8."""
    if isinstance(images, np.ndarray):
        tensor = torch.from_numpy(images)
    elif isinstance(images, torch.Tensor):
        tensor = images
    else:
        raise TypeError("images must be a NumPy array or torch.Tensor")

    if tensor.ndim != 5:
        raise ValueError("Expected stored image shape [3, N, 512, 512, 3].")
    if tensor.shape[0] != 3 or tensor.shape[2:] != (PATCH_SIZE, PATCH_SIZE, 3):
        raise ValueError("Expected stored image shape [3, N, 512, 512, 3].")
    if tensor.dtype != torch.uint8:
        raise ValueError("Stored augmentation images must have dtype uint8.")

    tensor = (
        tensor.to(device="cpu").permute(0, 1, 4, 2, 3).contiguous().to(torch.float32).div_(255.0)
    )
    mean, std = _normalization_tensors(device=tensor.device)
    return tensor.sub(mean).div(std)


def validate_augmented_hdf5(hdf5_path, *, expected=None, verify_values=True):
    """Validate the on-disk augmentation contract and return its metadata.

    ``expected`` may contain ``slide_id``, ``patient_id``, ``label``,
    ``patch_count`` or ``run_fingerprint``.  A mismatch is treated as
    corruption/stale output rather than silently returning misaligned data.
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Augmentation HDF5 not found: {hdf5_path}")
    expected = {} if expected is None else dict(expected)
    try:
        with h5py.File(hdf5_path, "r") as file:
            missing = REQUIRED_DATASETS.difference(file.keys())
            if missing:
                raise ValueError(
                    "Augmentation HDF5 is missing datasets: " + ", ".join(sorted(missing))
                )
            images = file["images"]
            coords = file["coords"]
            tissue = file["tissue_percentage"]
            if images.ndim != 5 or images.shape[0] != len(REPRESENTATIONS):
                raise ValueError("images must have shape [3,N,512,512,3].")
            if images.shape[2:] != (PATCH_SIZE, PATCH_SIZE, 3):
                raise ValueError("images must have shape [3,N,512,512,3].")
            if images.dtype != np.dtype(np.uint8):
                raise ValueError("images must have dtype uint8.")
            patch_count = int(images.shape[1])
            if coords.shape != (patch_count, 2) or coords.dtype != np.dtype(np.int64):
                raise ValueError("coords must have dtype int64 and shape [N,2].")
            if tissue.shape != (patch_count,) or tissue.dtype != np.dtype(np.float32):
                raise ValueError("tissue_percentage must have dtype float32 and shape [N].")

            required_attrs = {
                "format_version",
                "slide_id",
                "patient_id",
                "label",
                "representations",
                "image_layout",
                "stored_dtype",
                "restored_tensor_layout",
                "restored_tensor_dtype",
                "imagenet_mean",
                "imagenet_std",
                "patch_size",
                "patch_count",
            }
            missing_attrs = required_attrs.difference(file.attrs.keys())
            if missing_attrs:
                raise ValueError(
                    "Augmentation HDF5 is missing attributes: " + ", ".join(sorted(missing_attrs))
                )
            format_version = str(file.attrs["format_version"])
            if format_version not in SUPPORTED_FORMAT_VERSIONS:
                raise ValueError(f"Unsupported augmentation format_version: {format_version}")
            if int(file.attrs["patch_size"]) != PATCH_SIZE:
                raise ValueError("HDF5 patch_size does not match the runtime contract.")
            if int(file.attrs["patch_count"]) != patch_count:
                raise ValueError("HDF5 patch_count attribute does not match datasets.")
            if str(file.attrs["image_layout"]) != "representation,patch,height,width,channel":
                raise ValueError("Unexpected stored image layout.")
            if str(file.attrs["stored_dtype"]) != "uint8":
                raise ValueError("Unexpected stored dtype declaration.")
            if (
                str(file.attrs["restored_tensor_layout"])
                != "representation,patch,channel,height,width"
            ):
                raise ValueError("Unexpected restored tensor layout.")
            if str(file.attrs["restored_tensor_dtype"]) != "float32":
                raise ValueError("Unexpected restored tensor dtype declaration.")
            if (
                tuple(_decode_json_attribute(file.attrs["representations"], "representations"))
                != REPRESENTATIONS
            ):
                raise ValueError("Unexpected representation order.")
            mean = tuple(
                float(value)
                for value in _decode_json_attribute(file.attrs["imagenet_mean"], "imagenet_mean")
            )
            std = tuple(
                float(value)
                for value in _decode_json_attribute(file.attrs["imagenet_std"], "imagenet_std")
            )
            if mean != tuple(IMAGENET_MEAN) or std != tuple(IMAGENET_STD):
                raise ValueError("Stored ImageNet normalization metadata does not match runtime.")

            metadata = {
                "format_version": format_version,
                "slide_id": str(file.attrs["slide_id"]),
                "patient_id": str(file.attrs["patient_id"]),
                "label": str(file.attrs["label"]),
                "patch_count": patch_count,
                "run_fingerprint": str(file.attrs.get("run_fingerprint", "")),
            }
            for key, expected_value in expected.items():
                if key not in metadata:
                    raise ValueError(f"Unsupported expected metadata key: {key}")
                observed = metadata[key]
                if key == "patch_count":
                    expected_value = int(expected_value)
                else:
                    expected_value = str(expected_value)
                if observed != expected_value:
                    raise ValueError(
                        f"HDF5 metadata mismatch for {key}: "
                        f"expected {expected_value!r}, observed {observed!r}."
                    )

            if verify_values and patch_count:
                coordinates = coords[:]
                tissue_values = tissue[:]
                _validate_coordinate_and_tissue_values(coordinates, tissue_values)
            return metadata
    except OSError as error:
        raise ValueError(f"Cannot open augmentation HDF5: {hdf5_path}") from error


def load_augmented_chunk(hdf5_path, start=0, stop=None):
    """Load one saved augmentation chunk for direct KimiaNet consumption.

    Returns ``(transformed, coordinates, tissue_percentage)`` where transformed
    is float32 [3,B,3,512,512], coordinates is int64 [B,2], and tissue
    percentage is float32 [B].
    """
    hdf5_path = Path(hdf5_path)
    metadata = validate_augmented_hdf5(hdf5_path, verify_values=False)
    with h5py.File(hdf5_path, "r") as file:
        total = metadata["patch_count"]
        if stop is None:
            stop = total
        start = int(start)
        stop = int(stop)
        if start < 0 or stop < start or stop > total:
            raise ValueError(f"Invalid slice [{start}:{stop}] for {total} stored patches.")
        images = file["images"][:, start:stop]
        coordinates = np.asarray(file["coords"][start:stop], dtype=np.int64)
        tissue_percentage = np.asarray(
            file["tissue_percentage"][start:stop],
            dtype=np.float32,
        )

    _validate_coordinate_and_tissue_values(coordinates, tissue_percentage)

    transformed = uint8_views_to_normalized(images)
    return transformed, coordinates, tissue_percentage
