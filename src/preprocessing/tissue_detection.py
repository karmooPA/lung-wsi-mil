import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk


def image_to_array(image):
    image_array = np.asarray(image)
    return image_array


def rgb_to_gray(image_array):
    gray_image = rgb2gray(image_array)
    return gray_image

def create_tissue_mask(gray_image):
    threshold = threshold_otsu(gray_image)
    tissue_mask = gray_image < threshold
    
    return tissue_mask,threshold

def clean_tissue_mask(tissue_mask, radius=3):
    footprint = disk(radius)

    cleaned_mask = closing(
    tissue_mask,
    footprint=footprint
)

    return cleaned_mask

def get_average_tissue_color(image_array, tissue_mask):
    tissue_pixels = image_array[tissue_mask]

    average_color = tissue_pixels.mean(axis=0)

    return average_color

def calculate_color_distance(image_array, average_color):
    image_float = image_array.astype(np.float32)

    difference = image_float - average_color

    color_distance = np.linalg.norm(
        difference,
        axis=2
    )

    return color_distance

def filter_by_color_distance(
    tissue_mask,
    color_distance,
    threshold=35
):
    filtered_mask = tissue_mask & (color_distance < threshold)
    return filtered_mask