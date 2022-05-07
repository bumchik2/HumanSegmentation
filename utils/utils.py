import cv2
import numpy as np
from PIL import Image


def read_image(image_path: str) -> np.ndarray:
    """Returns a digital rgb representation of the image by path.
    Parameters
    ----------
    image_path
        Path to image.
    Returns
    -------
    np.ndarray
        3 x height x width rgb image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(mask_path: str) -> np.ndarray:
    """Returns a digital representation of mask by path.
    Parameters
    ----------
    mask_path
        Path to mask.
    Returns
    -------
    np.ndarray
        Height x width mask.
    """
    mask = np.array(Image.open(mask_path))
    assert(len(mask.shape) == 2)
    return mask
