# metrics/dice_coefficient.py
import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice coefficient, a measure of set similarity.
    
    Parameters:
    y_true (array): Ground truth binary labels.
    y_pred (array): Predicted binary labels.
    smooth (float): Small constant added to numerator and denominator for numerical stability.
    
    Returns:
    float: Dice coefficient.
    """
    # Flatten arrays to convert them into 1D vectors
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    
    # Calculate intersection and union
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

# Example usage (with dummy data):
# y_true = np.random.randint(0, 2, size=(128, 128, 128))
# y_pred = np.random.randint(0, 2, size=(128, 128, 128))
# print(dice_coefficient(y_true, y_pred))
