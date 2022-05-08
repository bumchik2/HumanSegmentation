from torch import nn
import torch


class ReflectionAggregator(nn.Module):
    """Reflection aggregator calculates the mask not only
    for the original image, but also for the image reflected horizontally.
    Then the reflected mask is reflected back and averaged with the mask
    obtained for the original image.
    """
    def __init__(self, model):
        """Initializes ReflectionAggregator.
        Parameters
        ----------
        model : torch.nn.Module
            The model that the aggregator wraps.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)

        x_reflected = torch.flip(x, dims=(-1,))
        y_reflected = self.model(x_reflected)

        return (y + torch.flip(y_reflected, dims=(-1,))) / 2
