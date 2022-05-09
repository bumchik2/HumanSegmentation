import torch
from utils.mask_utils import classification_border, binarize_mask
from lib.metrics import get_dice
from training.average_meter import AverageMeter
import numpy as np


@torch.no_grad()
def get_mask_prediction(model, image, transform, device='cuda') -> np.ndarray:
    """Predicts a segmentation mask for a specific image.
    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model.
    image : torch.tensor or np.ndarray, 3d
        Original image for segmentation.
    transform
        Transform applied to image before passing to model.
    device : str, optional (default='cuda')
        Device for computing (cuda or cpu).
    Returns
    -------
    np.ndarray, 2d
        Height x width mask with float values.
    """
    model.train(False)

    if transform is not None:
        image = transform(image)

    result = model(torch.unsqueeze(image, 0).to(device)).cpu().numpy()[0, 0]
    assert(len(result.shape) == 2)  # height x width
    return result


@torch.no_grad()
def get_metrics(model, criterion, val_batch_gen,
                border=classification_border, device='cuda') -> dict:
    """Calculates metrics (loss and dice) of the model.
    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model.
    criterion
        Pytorch criterion for computing loss.
    val_batch_gen : torch.utils.data.DataLoader
        Data loader for metrics evaluation.
    border : float
        Segmentation mask binarization border.
    device : str
        Device for computing (cuda or cpu).
    Returns
    -------
    Dict
        {'loss': average_loss, 'dice': average_dice}
    """
    model.train(False)

    val_loss = AverageMeter()
    val_dice = AverageMeter()
    for i, (x_batch, y_batch) in enumerate(val_batch_gen):
        x_batch = x_batch.to(device)  # (batch_size, 3, image_height, image_width)
        y_batch = y_batch.to(device)  # (batch_size, 1, image_height, image_width)

        y_pred = model(x_batch)

        val_loss_part = criterion(y_pred, y_batch)

        val_loss.update(val_loss_part.item(), len(x_batch))
        y_pred = binarize_mask(y_pred.detach().cpu().numpy(), border)
        y_batch = y_batch.cpu().numpy()
        val_dice_part = get_dice(y_batch, y_pred)
        val_dice.update(val_dice_part, len(x_batch))

    return {'loss': val_loss.avg, 'dice': val_dice.avg}
