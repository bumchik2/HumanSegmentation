"""Common UNet blocks.
"""


from torch import nn
import torch
import torch.nn.functional as F


class UNetDown(nn.Module):
    """Downsampling UNet block
    """
    def __init__(self, conv_block_class, in_channels, out_channels):
        super().__init__()

        self.down_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_block_class(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_block(x)


class UNetUp(nn.Module):
    """Upsampling UNet block
    """
    def __init__(self, conv_block_class, in_channels, out_channels):
        super().__init__()

        self.up_block = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = conv_block_class(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_block(x1)

        dheight = x2.size()[2] - x1.size()[2]
        dwidth = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [dwidth // 2, dwidth - dwidth // 2,
                        dheight // 2, dheight - dheight // 2])

        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """Common class for UNet architectures.
    """
    def __init__(self, conv_block_class, in_channels, out_channels):
        """Initializes common class for UNet architectures.
        Parameters
        ----------
        conv_block_class : torch.nn.Module
            The class used for convolutions that do not change the size of the image.
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the model.
        """
        super().__init__()

        self.inc = conv_block_class(in_channels, 64)

        self.down1 = UNetDown(conv_block_class, 64, 128)
        self.down2 = UNetDown(conv_block_class, 128, 256)
        self.down3 = UNetDown(conv_block_class, 256, 512)
        self.down4 = UNetDown(conv_block_class, 512, 1024)

        self.up1 = UNetUp(conv_block_class, 1024, 512)
        self.up2 = UNetUp(conv_block_class, 512, 256)
        self.up3 = UNetUp(conv_block_class, 256, 128)
        self.up4 = UNetUp(conv_block_class, 128, 64)

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.last_conv(x)
        return x
