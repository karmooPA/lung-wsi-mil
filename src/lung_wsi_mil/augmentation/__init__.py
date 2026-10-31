"""Three aligned representations of each 512x512 RGB patch."""

from .data_augmentations import transform_batch, transform_patch

__all__ = ["transform_patch", "transform_batch"]
