from torch import nn
import torch
import torch.nn.functional as F


class SoftErosion(nn.Module):
    """Applies a convolution with a fixed kernel for segmentation masks smoothing.
    """
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        """Initializes SoftErosion.
        Parameters
        ----------
        kernel_size : int
            Size of the convolving kernel.
        threshold : float
            The values that will be above this value after applying the convolution will be set to 1.
        iterations : int
            Number of times the convolution is applied.
        """
        super().__init__()

        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()

        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[0], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[0], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x


@torch.no_grad()
def get_smooth_mask(mask, mask_smoother, device):
    """Applies a mask smoother model to the mask
    Parameters
    ----------
    mask : np.ndarray, 2d
        Segmentation_mask height x width.
    mask_smoother : float
        Model for smoothing masks.
    device : str
        Device for computing (cuda or cpu).
    Returns
    -------
    np.ndarray, 2d
        Smoothed mask.
    """
    mask = torch.tensor(mask).to(device)
    mask = mask.unsqueeze_(0)
    result = mask_smoother(mask)[0].cpu().numpy()
    return result
