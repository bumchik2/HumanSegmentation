import numpy as np


binarization_border = 0.5


def binarize_mask(mask, border=binarization_border):
    """Binarizes segmentation mask.
    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask height x width or mutiple masks batch_size x height x width.
    border : float
        Border of binarization.
    Returns
    -------
    np.ndarray
        Binarized mask.
    """
    return (mask > border).astype(int)
