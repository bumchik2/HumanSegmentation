from utils.mask_utils import classification_border, binarize_mask
from training.average_meter import AverageMeter
from tqdm import tqdm
from lib.metrics import get_dice
import numpy as np
import time
from collections import defaultdict
from inference.model_inference import get_metrics
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
from plots.plot_masks import plot_masks_comparisons
from typing import List, Dict


def print_metrics(history: Dict[str, Dict[str, List[float]]]):
    """Prints loss and dice on train and validation samples into standard output.
    Parameters
    ----------
    history
        Object with training progress information.
        history['loss' or 'dice']['train' or 'val'] should return list of float.
    """

    for metric in ('loss', 'dice'):
        for sample_name in ('train', 'val'):
            print(f'{sample_name} {metric}: {history[metric][sample_name][-1]}')


def wandb_log_metrics(history: Dict[str, Dict[str, List[float]]]):
    """Logs metrics to wandb. Generates plots for loss and dice on train and validation samples.
    Parameters
    ----------
    history
        Object with training progress information.
        history['loss' or 'dice']['train' or 'val'] should return list of float.
    """

    import wandb

    epochs_passed = len(history['loss']['train'])

    for metric in ('loss', 'dice'):
        for sample_name in ('train', 'val'):
            wandb.log({f'{metric}_{sample_name}': history[metric][sample_name][-1]})

    for metric in ('loss', 'dice'):
        wandb.log({metric: wandb.plot.line_series(
            xs=range(epochs_passed),
            ys=[history[metric]['train'], history[metric]['val']],
            keys=['train', 'val'],
            title=metric,
            xname='Epoch'
        )})


def plot_learning_curves(history: Dict[str, Dict[str, List[float]]]):
    """Draws model training graphs.
    Parameters
    ----------
    history
        Object with training progress information.
        history['loss' or 'dice']['train' or 'val'] should return list of float.
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('Loss', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 2, 2)
    plt.title('Среднее значение Dice', fontsize=15)
    plt.plot(history['dice']['train'], label='train')
    plt.plot(history['dice']['val'], label='val')
    plt.ylabel('Dice', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend()
    plt.grid(visible=True)
    plt.show()


def train_one_epoch(model: torch.nn.Module, criterion, optimizer, train_batch_gen: torch.utils.data.DataLoader,
                    border: float = classification_border, device='cuda') -> dict:
    """Performs one epoch of training segmentation model.
    Parameters
    ----------
    model
        Segmentation model to train.
    criterion
        Pytorch criterion for computing loss.
    optimizer
        One of torch.optim optimizers: https://pytorch.org/docs/stable/optim.html .
    train_batch_gen
        Data loader for training.
    border
        Segmentation mask binarization border.
    device
        Device for computing (cuda or cpu).
    Returns
    -------
        Dict {'loss': epoch_average_train_loss, 'dice': epoch_average_train_dice}
    """
    model.train(True)

    train_loss = AverageMeter()
    train_dice = AverageMeter()
    for i, (x_batch, y_batch) in tqdm(enumerate(train_batch_gen),
                                      total=len(train_batch_gen), position=0, leave=True):
        x_batch = x_batch.to(device)  # (batch_size, 3, image_height, image_width)
        y_batch = y_batch.to(device)  # (batch_size, 1, image_height, image_width)

        y_pred = model(x_batch)

        train_loss_part = criterion(y_pred, y_batch)
        train_loss_part.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.update(train_loss_part.item(), len(x_batch))
        y_pred = binarize_mask(y_pred.detach().cpu().numpy(), border)
        y_batch = y_batch.cpu().numpy()
        train_dice_part = get_dice(y_batch, y_pred)
        train_dice.update(train_dice_part, len(x_batch))

    return {'loss': train_loss.avg, 'dice': train_dice.avg}


def train_model(model, criterion, optimizer, train_batch_gen, val_batch_gen, num_epochs, save_prefix, test_transform,
                images_paths, masks_paths, scheduler, use_wandb=True, device='cuda', border=classification_border):
    """Trains model, evaluates metrics and draws learning curves. Saves the best model to save_prefix + '_best.pt'.
    Saves the latest model to save_prefix + '_latest.pt'. If use_wandb is set to True,
    training logs will be sent to wandb (see datasets.wandb_log_metrics).
    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model to train.
    criterion
        Pytorch criterion for computing loss.
    optimizer
        One of torch.optim optimizers: https://pytorch.org/docs/stable/optim.html .
    train_batch_gen : torch.utils.data.DataLoader
        Data loader for training.
    val_batch_gen : torch.utils.data.DataLoader
        Data loader for metrics evaluation.
    num_epochs : int
        Number of epochs to train.
    save_prefix : str
        Prefix for path to save best and latest models. The best model (by validation loss)
        will be saved into save_prefix + '_best.pt', the latest - into save_prefix + '_latest.pt'
    test_transform
        Transform applied to images, which are tested every epoch before passing into model.
    images_paths: : List[str]
        Image paths for to be tested every epoch during training.
    masks_paths : List[str]
        Real segmentation masks of the images tested every epoch during training.
    scheduler
        Pytorch scheduler for the optimizer.
    use_wandb : bool
        Whether or not to use wandb for logging. Set to True by default.
    device : str
        Device for computing (cuda or cpu).
    border : float
        Segmentation mask binarization border.
    Returns
    ----------
    model
        The trained model.
    history
        Object with training progress information.
        history['loss' or 'dice']['train' or 'val'] will is list of float.
    """

    if use_wandb:
        import wandb

    assert (len(images_paths) == len(masks_paths))
    history = defaultdict(lambda: defaultdict(list))
    best_val_metric = np.inf

    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(model, criterion, optimizer, train_batch_gen,
                                        border=border, device=device)
        history['loss']['train'].append(train_metrics['loss'])
        history['dice']['train'].append(train_metrics['dice'])

        if scheduler is not None:
            scheduler.step()

        val_metrics = get_metrics(model, criterion, val_batch_gen,
                                  border=border, device=device)
        history['loss']['val'].append(val_metrics['loss'])
        history['dice']['val'].append(val_metrics['dice'])

        # Сохраняем модель
        val_metric = val_metrics['loss']
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), save_prefix + '_best.pt')
        torch.save(model.state_dict(), save_prefix + '_latest.pt')

        # Выводим метрики, строим графики
        clear_output()
        print(f'Epoch {epoch + 1} of {num_epochs} took {round(time.time() - epoch_start_time, 3)} seconds')
        print_metrics(history)
        plot_learning_curves(history)
        plot_masks_comparisons(model, test_transform, images_paths, masks_paths, device=device)
        if use_wandb:
            wandb_log_metrics(history)

    if use_wandb:
        wandb.finish()

    return model, history
