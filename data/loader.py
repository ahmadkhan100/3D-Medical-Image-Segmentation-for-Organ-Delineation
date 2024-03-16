# data/loader.py
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

def load_nifti_file(filepath):
    """
    Load a NIfTI file and return it as a numpy array.
    """
    nifti_img = nib.load(filepath)
    return nifti_img.get_fdata()

def resize_volume(img, desired_shape):
    """
    Resize a 3D volume to desired shape.
    """
    current_shape = img.shape
    resize_factor = np.array(desired_shape) / np.array(current_shape)
    resized_img = zoom(img, resize_factor, mode='nearest')
    return resized_img

def standardize_volume(img):
    """
    Standardize the volume by setting its mean to 0 and variance to 1.
    """
    standardized_img = (img - np.mean(img)) / np.std(img)
    return standardized_img

def normalize_volume(img):
    """
    Normalize the volume to be between 0 and 1.
    """
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

def preprocess_volume(img, desired_shape):
    """
    Full preprocessing pipeline to load, resize, and normalize a volume.
    """
    img = load_nifti_file(img)
    img = resize_volume(img, desired_shape)
    img = standardize_volume(img)
    return img

def preprocess_data(image_paths, label_paths, desired_shape):
    """
    Load, preprocess, and split the image and label data.
    """
    images = [preprocess_volume(img_path, desired_shape) for img_path in image_paths]
    labels = [preprocess_volume(label_path, desired_shape) for label_path in label_paths]
    
    # Split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)
    
    return np.array(train_images), np.array(train_labels), np.array(val_images), np.array(val_labels)

# Here we assume that your images are already in memory and you have their file paths.
# Replace 'desired_shape' with the actual shape you need for your model.
image_paths = [...]  # List of file paths for your images
label_paths = [...]  # List of file paths for your labels
desired_shape = (128, 128, 128)  # Example shape

train_images, train_labels, val_images, val_labels = preprocess_data(image_paths, label_paths, desired_shape)
