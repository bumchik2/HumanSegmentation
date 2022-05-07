import numpy as np


classification_border = 0.5


def binarize_mask(mask: np.ndarray, border: float = 0.5) -> np.ndarray:
    """Binarizes segmentation mask.
    Parameters
    ----------
    mask
        Segmentation_mask height x width.
    border
        Border of binarization.
    Returns
    -------
        Binarized mask.
    """
    return (mask > border).astype(int)
