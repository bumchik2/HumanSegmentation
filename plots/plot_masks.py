import matplotlib.pyplot as plt
from utils.utils import read_image, read_mask
from utils.mask_utils import binarize_mask, classification_border
from inference.model_inference import get_mask_prediction
import numpy as np
import torch
from typing import List


def plot_masks_comparison(image: np.ndarray, real_mask: np.ndarray,
                          predicted_mask: np.ndarray, border: float = classification_border):
    """Visualizes real and predicted mask
    Parameters
    ----------
    image
        Original image for segmentation.
    real_mask
        Real segmentation mask.
    predicted_mask:
        Predicted segmentation mask.
    border:
        Segmentation mask binarization border.
    """
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение", fontsize=14)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Реальная маска", fontsize=14)
    plt.imshow(real_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Предсказанная маска", fontsize=14)
    plt.imshow(binarize_mask(predicted_mask, border))
    plt.axis("off")
    plt.show()


def plot_masks_comparisons(model: torch.nn.Module, test_transform, images_paths: List[str],
                           masks_paths: List[str], border: float = classification_border, device: str = 'cuda'):
    """Predicts masks for given image paths,
    draws original images, real and predicted masks.
    Parameters
    ----------
    model
        Segmentation model.
    test_transform
        Transform applied to images before passing into model.
    images_paths:
        Image paths for segmentation.
    masks_paths:
        Real segmentation masks.
    border:
        Segmentation mask binarization border.
    device:
        Device for computing (cuda or cpu).
    """
    for image_path, mask_path in zip(images_paths, masks_paths):
        image = read_image(image_path)
        real_mask = read_mask(mask_path)
        predicted_mask = get_mask_prediction(model, image, test_transform, device=device)
        plot_masks_comparison(image, real_mask, predicted_mask, border)
