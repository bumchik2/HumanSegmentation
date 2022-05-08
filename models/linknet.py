"""The original article about LinkNet: https://arxiv.org/pdf/1707.03718.pdf
Below is a slightly modified version of the model:
the authors of the article worked with high-resolution images (640x360),
so I removed downscaling at the beginning and, accordingly, upscaling at the end.
I also slightly altered the EncoderBlock, leaving only one residual link there.
"""


from torch import nn


class LinkNetEncoderBlock(nn.Module):
    """LinkNet encoder block. Consists of 3 convolutions with 1 residual connection.
    Does x2 downsampling over the incoming image.
    """
    def __init__(self, m, n):
        """LinkNet encoder block. Consists of 3 convolutions with 1 residual connection.
        Does x2 downsampling over the incoming image.
        Parameters
        ----------
        m : int
            Number of channels in the input image.
        n : int
            Number of channels produced by the encoder block.
        """
        super().__init__()

        # residual block
        self.block1 = nn.Sequential(
            nn.Conv2d(m, m, kernel_size=3, padding=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, m, kernel_size=3, padding=1),
            nn.BatchNorm2d(m),
            nn.ReLU()
        )

        # x2 downsampling block
        self.block2 = nn.Sequential(
            nn.Conv2d(m, n, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(n),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.block1(x)
        return self.block2(x)


class LinkNetDecoderBlock(nn.Module):
    """LinkNet decoder block. Convolution + deconvolution + convolution with batchnorm + relu in between.
    Does x2 upsampling over the incoming image.
    """
    def __init__(self, m, n):
        """LinkNet decoder block. Convolution + deconvolution + convolution with batchnorm + relu in between.
        Does x2 upsampling over the incoming image.
        Parameters
        ----------
        m : int
            Number of channels in the input image.
        n : int
            Number of channels produced by the encoder block.
        """
        super().__init__()

        # x2 upsampling block
        self.block = nn.Sequential(
            nn.Conv2d(m, m // 4, kernel_size=1),
            nn.BatchNorm2d(m // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(m // 4, m // 4, stride=2, kernel_size=3,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(m // 4),
            nn.ReLU(),
            nn.Conv2d(m // 4, n, kernel_size=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LinkNet(nn.Module):
    """LinkNet with 4 encoder blocks and 4 decoder blocks.
    """
    def __init__(self, in_channels, out_channels):
        """LinkNet with 4 encoder blocks and 4 decoder blocks.
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the model.
        """
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder1 = LinkNetEncoderBlock(64, 64)
        self.encoder2 = LinkNetEncoderBlock(64, 128)
        self.encoder3 = LinkNetEncoderBlock(128, 256)
        self.encoder4 = LinkNetEncoderBlock(256, 512)

        self.decoder1 = LinkNetDecoderBlock(64, 64)
        self.decoder2 = LinkNetDecoderBlock(128, 64)
        self.decoder3 = LinkNetDecoderBlock(256, 128)
        self.decoder4 = LinkNetDecoderBlock(512, 256)

        self.finish = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.start(x)

        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(encoder1_output)
        encoder3_output = self.encoder3(encoder2_output)
        encoder4_output = self.encoder4(encoder3_output)

        decoder4_output = self.decoder4(encoder4_output)
        decoder3_output = self.decoder3(decoder4_output + encoder3_output)
        decoder2_output = self.decoder2(decoder3_output + encoder2_output)
        decoder1_output = self.decoder1(decoder2_output + encoder1_output)

        return self.finish(decoder1_output)
