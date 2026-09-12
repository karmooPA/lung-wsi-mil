import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing


def image_to_array(image):
    """Convert a PIL image to a NumPy array."""
    return np.asarray(image)


def rgb_to_gray(image_array):
    """Convert an RGB NumPy image to grayscale in the [0, 1] range."""
    return rgb2gray(image_array)


def create_tissue_mask(gray_image):
    """Detect dark tissue with Otsu thresholding."""
    threshold = threshold_otsu(gray_image)
    tissue_mask = gray_image < threshold
    return tissue_mask, threshold


def clean_tissue_mask(tissue_mask):
    """
    Morphologically clean the tissue mask with a 3x3 footprint.

    Important reference-detail:
    WSI_MIL_ROI applies erosion followed by dilation to an OpenCV binary image
    where tissue is BLACK and background is WHITE. In our representation tissue
    is True/foreground, so the equivalent tissue operation is closing, not
    opening. Keeping this polarity explicit avoids accidentally changing the
    reference behaviour.
    """
    footprint = np.ones((3, 3), dtype=bool)
    return closing(tissue_mask, footprint=footprint)


def get_average_tissue_color(image_array, tissue_mask):
    """Return mean RGB color of pixels selected as tissue."""
    tissue_pixels = image_array[tissue_mask]

    if tissue_pixels.size == 0:
        return None

    return tissue_pixels.mean(axis=0)


def calculate_color_distance(image_array, average_color):
    """Calculate per-pixel Euclidean RGB distance from average tissue color."""
    if average_color is None:
        return np.full(image_array.shape[:2], np.inf, dtype=np.float32)

    image_float = image_array.astype(np.float32)
    difference = image_float - average_color
    return np.linalg.norm(difference, axis=2)


def filter_by_color_distance(
    tissue_mask,
    color_distance,
    threshold=35,
):
    """Keep tissue pixels whose RGB distance is below the reference threshold."""
    return tissue_mask & (color_distance < threshold)
