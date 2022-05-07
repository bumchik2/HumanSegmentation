"""HRNet from https://arxiv.org/pdf/2001.05566.pdf
The main idea is to maintain several convolution sequences at different resolutions simultaneously.
In my implementation the information between these sequences is mixed using HRFusion blocks.
"""


from torch import nn
import torch
from models.common import ConvSequence, Conv3x3, ConvDownSample, ConvUpSample


class HRFusion(nn.Module):
    def __init__(self, in_channels, out_channels_each, in_downsampling_factors, out_downsampling_factors):
        super().__init__()

        convs = []

        self.in_downsampling_factors = in_downsampling_factors
        self.out_downsampling_factors = out_downsampling_factors

        for out_downsampling_factor in out_downsampling_factors:
            for in_downsampling_factor in in_downsampling_factors:
                relative_downsampling_factor = out_downsampling_factor / in_downsampling_factor
                if relative_downsampling_factor >= 1:
                    convs.append(ConvDownSample(in_channels, out_channels_each, round(relative_downsampling_factor)))
                else:
                    convs.append(ConvUpSample(in_channels, out_channels_each, round(1 / relative_downsampling_factor)))

        self.convs = nn.ModuleList(
            convs
        )

    def forward(self, x):
        # x has to be of shape
        # batch_size x tuple(in_channels x height_0 x width_0, in_channels x height_1 x width_1, ...)

        result = []

        for i, out_downsampling_factor in enumerate(self.out_downsampling_factors):
            result_i = []

            for j, in_downsampling_factor in enumerate(self.in_downsampling_factors):
                index = i * len(self.in_downsampling_factors) + j
                result_i_part = self.convs[index](x[j])
                result_i.append(result_i_part)

            result_i = torch.cat(result_i, dim=1)
            # result_i has shape
            # batch_size x (out_channels_each * len(in_downsampling_factors)) x height_i x width_i

            result.append(result_i)

        return result


class HRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.start = nn.Sequential(
            ConvDownSample(in_channels, 16, 2),
            ConvSequence(16, 32)
        )

        self.fusion1 = HRFusion(32, 32, (1,), (1, 2))

        self.conv2 = ConvSequence(32, 64)
        self.conv3 = ConvSequence(32, 64)
        self.fusion2 = HRFusion(64, 64, (1, 2), (1, 2, 4))

        self.conv4 = ConvSequence(128, 128)
        self.conv5 = ConvSequence(128, 128)
        self.conv6 = ConvSequence(128, 128)
        self.fusion3 = HRFusion(128, 32, (1, 2, 4), (1, 2, 4, 8))

        self.conv7 = ConvSequence(96, 64)
        self.conv8 = ConvSequence(96, 64)
        self.conv9 = ConvSequence(96, 64)
        self.conv10 = ConvSequence(96, 64)

        self.fusion4 = HRFusion(64, 16, (1, 2, 4, 8), (1, 2, 4, 8))

        self.fusion5 = HRFusion(64, 16, (1, 2, 4, 8), (1,))

        self.finish = nn.Sequential(
            ConvUpSample(64, 32, 2),
            Conv3x3(32, out_channels)
        )

    def forward(self, x):
        x = self.start(x)

        x, y = self.fusion1((x,))

        x = self.conv2(x)
        y = self.conv3(y)
        x, y, z = self.fusion2((x, y,))

        x = self.conv4(x)
        y = self.conv5(y)
        z = self.conv6(z)
        x, y, z, w = self.fusion3((x, y, z,))

        x = self.conv7(x)
        y = self.conv8(y)
        z = self.conv9(z)
        w = self.conv10(w)
        x, y, z, w = self.fusion4((x, y, z, w,))

        x = self.fusion5((x, y, z, w))[0]
        x = self.finish(x)
        return x
