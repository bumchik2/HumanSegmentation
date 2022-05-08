from torch import nn
from copy import deepcopy


class MaskTuner(nn.Module):
    """A wrapper that adds several additional convolutional layers.
    The main model remains frozen and is not trained.
    """
    def __init__(self, model):
        """Initializes MaskTuner.
        Parameters
        ----------
        model : torch.nn.Module
            The model that the mask tuner wraps.
        """
        super().__init__()
        self.model = deepcopy(model)
        for param in self.model.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=7)

    def forward(self, x):
        x = self.model(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.deconv1(y)
        y = self.deconv2(y)
        y = x + y
        return y
