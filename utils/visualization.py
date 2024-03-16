# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, title="Image"):
    """
    Plot a 2D medical image.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image_mask(image, mask, title="Image and Mask"):
    """
    Plot a 2D medical image and overlay a segmentation mask.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage:
# plot_image(sample_image)
# plot_image_mask(sample_image, sample_mask)

