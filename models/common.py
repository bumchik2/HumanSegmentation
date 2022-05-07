"""Typical constructions that can be used in many CNNs.
"""


from torch import nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv3x3(in_channels, in_channels),
            Conv3x3(in_channels, out_channels),
            Conv3x3(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class ConvDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        # downsample_factor has to be a power of 2
        assert(downsample_factor & (downsample_factor - 1) == 0)

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=downsample_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor):
        # upsample_factor has to be a power of 2
        assert(upsample_factor & (upsample_factor - 1) == 0)

        super().__init__()

        output_padding = upsample_factor - 1
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, stride=upsample_factor,
                               kernel_size=3, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.deconv(x)