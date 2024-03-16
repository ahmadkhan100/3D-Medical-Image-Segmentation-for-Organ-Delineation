# data/augmentation.py
import numpy as np
import scipy.ndimage

def random_rotation_3d(image, max_angle):
    """Rotate the image by a random angle within the given range."""
    angle = np.random.uniform(-max_angle, max_angle)
    rotated_image = scipy.ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
    return rotated_image

def random_flip_3d(image, axis=None):
    """Flip the image randomly along a specified axis."""
    if axis is None:
        axis = np.random.choice([0, 1, 2])
    flipped_image = np.flip(image, axis=axis)
    return flipped_image

def random_zoom_3d(image, min_zoom=0.9, max_zoom=1.1):
    """Zoom into the image by a random factor within the given range."""
    zoom_factor = np.random.uniform(min_zoom, max_zoom)
    zoomed_image = scipy.ndimage.zoom(image, zoom_factor)
    # Cropping or padding if the image size changes
    if zoom_factor < 1.0:
        pad_width = ((image.shape[0] - zoomed_image.shape[0]) // 2,
                     (image.shape[1] - zoomed_image.shape[1]) // 2,
                     (image.shape[2] - zoomed_image.shape[2]) // 2)
        zoomed_image = np.pad(zoomed_image, pad_width, mode='constant')
    elif zoom_factor > 1.0:
        crop_start = ((zoomed_image.shape[0] - image.shape[0]) // 2,
                      (zoomed_image.shape[1] - image.shape[1]) // 2,
                      (zoomed_image.shape[2] - image.shape[2]) // 2)
        crop_end = (crop_start[0] + image.shape[0],
                    crop_start[1] + image.shape[1],
                    crop_start[2] + image.shape[2])
        zoomed_image = zoomed_image[crop_start[0]:crop_end[0],
                                    crop_start[1]:crop_end[1],
                                    crop_start[2]:crop_end[2]]
    return zoomed_image

def random_shift_3d(image, max_shift=10):
    """Shift the image by a random number of pixels in each direction."""
    shift = np.random.uniform(-max_shift, max_shift, 3)
    shifted_image = scipy.ndimage.shift(image, shift)
    return shifted_image

def add_random_noise_3d(image, noise_factor=0.1):
    """Add random noise to the image."""
    noise = np.random.normal(0, noise_factor, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Assuming the image is normalized
    return noisy_image

# You would use these functions within your data preprocessing pipeline, 
# ensuring that the augmentations are applied consistently to both your images and labels.
