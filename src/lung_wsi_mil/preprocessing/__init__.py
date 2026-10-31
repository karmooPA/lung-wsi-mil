"""WSI preprocessing utilities for the lung WSI MIL project."""

from .pipeline import preprocess_slide

PREPROCESSING_VERSION = "0.2.0"

__all__ = ["preprocess_slide", "PREPROCESSING_VERSION"]
