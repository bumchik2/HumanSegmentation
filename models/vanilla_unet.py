"""Vanilla UNet architecture.
"""

from models.unet import UNet
from torch import nn


class DoubleConvBlock(nn.Module):
    """Vanilla UNet double convolutional block.
    (conv 3x3 + batchnorm + relu) x 2
    """
    def __init__(self, in_channels, out_channels):
        """Vanilla UNet double convolutional block.
        (conv 3x3 + batchnorm + relu) x 2
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the block.
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class VanillaUNet(nn.Module):
    """VanillaUNet model.
    """
    def __init__(self, in_channels, out_channels):
        """VanillaUNet model.
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the model.
        """
        super().__init__()

        self.unet = UNet(DoubleConvBlock, in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)
