import cv2
import numpy as np
from PIL import Image


def read_image(image_path):
    """Returns a digital rgb representation of the image by path.
    Parameters
    ----------
    image_path : str
        Path to image.
    Returns
    -------
    np.ndarray, 3d
        3 x height x width rgb image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(mask_path):
    """Returns a digital representation of mask by path.
    Parameters
    ----------
    mask_path : str
        Path to mask.
    Returns
    -------
    np.ndarray, 2d
        Height x width mask.
    """
    mask = np.array(Image.open(mask_path))
    assert(len(mask.shape) == 2)
    return mask
