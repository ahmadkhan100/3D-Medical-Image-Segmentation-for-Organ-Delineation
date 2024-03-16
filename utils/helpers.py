# utils/helpers.py
import numpy as np
import cv2

def normalize_image(image):
    """
    Normalize image data to 0-1 range.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def load_image_from_file(file_path, color_mode=cv2.IMREAD_GRAYSCALE):
    """
    Load an image from a file.
    """
    image = cv2.imread(file_path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    return image

# Example usage:
# image = load_image_from_file('path/to/image.png')
# normalized_image = normalize_image(image)
