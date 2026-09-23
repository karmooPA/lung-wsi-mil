"""Task-PDF augmentation protocol; see docs/data-augmentation.md for deviations.

The historical name RandomGaussianNoise denotes BLUR, not additive noise.
Use a caller-owned numpy.random.Generator for reproducible independent draws.
No global NumPy/Python/PyTorch seed is modified.
"""

import numpy as np
from PIL import Image, ImageFilter
from skimage.color import hed2rgb, rgb2hed
import torch
from torchvision import transforms

PATCH_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomHEStain:
    """Multiply H and E by independent N(1, 0.02) scalars; keep D unchanged."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, image):
        hed = rgb2hed(np.asarray(image, dtype=np.float64) / 255.0)
        hed[..., :2] *= self.rng.normal(1.0, 0.02, size=2)
        rgb = np.clip(hed2rgb(hed), 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).astype(np.uint8))


class RandomGaussianBlur:
    """Gaussian blur with scalar radius |N(0, 0.5)| pixels.

    Absolute value makes the radius valid and retains its magnitude; tested
    Pillow 12.3 treats +/- scalar radii equally. The PDF's length-one ndarray
    is not a portable Pillow radius. This operation adds no pixel noise.
    """

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, image):
        radius = abs(float(self.rng.normal(0.0, 0.5)))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomGaussianNoise(RandomGaussianBlur):
    """Compatibility name from the task PDF; the operation is Gaussian blur."""


class RotateRandom:
    """Uniform quarter-turns, implemented without interpolating tissue pixels."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, image):
        turn = int(self.rng.integers(0, 4))
        if turn == 0:
            return image.copy()
        operation = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                     Image.Transpose.ROTATE_270)[turn - 1]
        return image.transpose(operation)


class RandomHorizontalFlip:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, image):
        if self.rng.random() < 0.5:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image.copy()


class RandomVerticalFlip:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, image):
        if self.rng.random() < 0.5:
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image.copy()


normal_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_aug_transforms(rng=None):
    """Two pipelines using distinct draws from one advancing random stream."""
    rng = np.random.default_rng() if rng is None else rng
    return [transforms.Compose([
        RandomHEStain(rng),
        RandomGaussianNoise(rng),
        RotateRandom(rng),
        RandomHorizontalFlip(rng),
        RandomVerticalFlip(rng),
        normal_transforms,
    ]) for _ in range(2)]


aug_transforms = make_aug_transforms()


def _validate_patch(patch):
    if not isinstance(patch, Image.Image):
        raise TypeError("patch must be a PIL.Image.Image from extract_patch")
    if patch.mode != "RGB" or patch.size != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("patch must be RGB with size (512, 512); no implicit resize or conversion")


def _views(patch, augmentations):
    return tuple(transform(patch).to(dtype=torch.float32)
                 for transform in (normal_transforms, *augmentations))


def transform_patch(patch, *, rng=None):
    """Return (Original, Aug1, Aug2), each CPU float32 [3, 512, 512].

    Pass np.random.default_rng(seed) once and reuse it across patches. Recreating
    the same seed for each patch repeats the random sequence and is discouraged.
    Independent draws can coincide; different pixel arrays are not guaranteed.
    """
    _validate_patch(patch)
    augmentations = aug_transforms if rng is None else make_aug_transforms(rng)
    return _views(patch, augmentations)


def transform_batch(patches, *, rng=None):
    """Return CPU float32 [3, N, 3, 512, 512], preserving input patch order.

    Empty input returns [3, 0, 3, 512, 512]. Allocate bounded batches: the output
    alone costs 9 MiB per patch. The three views always share the original WSI
    coordinate; the transform never changes the caller's coordinate list.
    """
    patches = list(patches)
    for patch in patches:
        _validate_patch(patch)
    augmentations = aug_transforms if rng is None else make_aug_transforms(rng)
    output = torch.empty((3, len(patches), 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
    for index, patch in enumerate(patches):
        for version, tensor in enumerate(_views(patch, augmentations)):
            output[version, index] = tensor
    return output
