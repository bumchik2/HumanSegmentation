"""Article about SDU-Net: https://arxiv.org/ftp/arxiv/papers/2004/2004.03466.pdf
I replaced the usual convolutional blocks in vanilla UNet with dilated ones
to increase the perceptive field in neurons.
The authors of the article claim that such an architecture allowed to outperform
state-of-the-art models on datasets with segmentation of medical images.
"""


from torch import nn
import torch
from models.unet import UNet


class DilatedConv(nn.Module):
    """Convolution with kernel 3x3 and dilation 2 + batchnorm + relu
    """
    def __init__(self, in_channels, out_channels):
        """Convolution with kernel 3x3 and dilation 2 + batchnorm + relu
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        """
        super().__init__()

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.dilated_conv(x)


class DilatedConvBlock(nn.Module):
    """SDUNet convolutional block.
    Stacks multiple dilated convolutions.
    """
    def __init__(self, in_channels, out_channels):
        """SDUNet convolutional block.
        Stacks multiple dilated convolutions.
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the block.
        """
        assert (out_channels % 16 == 0)

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.dilated_conv1 = DilatedConv(out_channels // 2, out_channels // 4)
        self.dilated_conv2 = DilatedConv(out_channels // 4, out_channels // 8)
        self.dilated_conv3 = DilatedConv(out_channels // 8, out_channels // 16)
        self.dilated_conv4 = DilatedConv(out_channels // 16, out_channels // 16)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.dilated_conv1(x)
        x2 = self.dilated_conv2(x1)
        x3 = self.dilated_conv3(x2)
        x4 = self.dilated_conv4(x3)
        return torch.cat([x, x1, x2, x3, x4], dim=1)


class SDUNet(nn.Module):
    """SDUNet model.
    Uses stacked dilated convolutions instead of vanilla double convolutional blocks.
    """
    def __init__(self, in_channels, out_channels):
        """SDUNet model.
        Uses stacked dilated convolutions instead of vanilla double convolutional blocks.
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the model.
        """
        super().__init__()

        self.unet = UNet(DilatedConv, in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)
