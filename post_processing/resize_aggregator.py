from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResizeAggregator(nn.Module):
    """Resize aggregator resizes the input image to different sizes,
    applies the model, and then averages the results over all sizes.
    """
    def __init__(self, model, sizes):
        """Initializes ResizeAggregator.
        Parameters
        ----------
        model : torch.nn.Module
            The model that the aggregator wraps.
        sizes : List
            Sizes to which the images will be resized.
        """
        super().__init__()
        self.model = deepcopy(model)
        self.sizes = sizes

    def forward(self, x):
        results = []
        initial_size = x.shape[-2:]
        for size in self.sizes:
            x_reshaped = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
            result_reshaped = self.model(x_reshaped)
            result = F.interpolate(result_reshaped, size=initial_size, mode='bilinear', align_corners=True)
            results.append(result)

        results = torch.stack(results)
        return torch.mean(results, axis=0)
