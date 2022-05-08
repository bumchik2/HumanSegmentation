import numpy as np


classification_border = 0.5


def binarize_mask(mask, border=0.5):
    """Binarizes segmentation mask.
    Parameters
    ----------
    mask : np.ndarray
        Segmentation_mask height x width.
    border : float
        Border of binarization.
    Returns
    -------
    np.ndarray
        Binarized mask.
    """
    return (mask > border).astype(int)
