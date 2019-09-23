import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    Layer,
    BatchNorm2d,
    Conv2d,
    Conv2dScaleChannels,
    PixelShuffle_ICNR,
    ResidualBlock,
    UnetBlockWide,
    concat,
)


class DynamicUnetWide(Layer):
    """Create a U-Net from a given architecture."""

    # Assume the input channels is 3
    input_channels = 3
    net_base_channels = 256

    def __init__(
        self,
        encoder,
        channels_factor=1,
        batchnorm=False,
        spectral_norm=False,
        input_batchnorm=False,
    ):
        """`encoder` must be a resnet"""
        super().__init__()
        self.channels_factor = channels_factor
        self.channels = self.net_base_channels * self.channels_factor
        self.input_batchnorm = input_batchnorm
        self.batchnorm = batchnorm
        self.spectral_norm = spectral_norm

        # Define the network
        self.batchnorm_in = BatchNorm2d() if self.input_batchnorm else lambda x: x
        self.encoder = encoder
        self.neck = nn.Sequential(
            Conv2dScaleChannels(
                scale=2,
                kernel_size=3,
                activation=nn.ReLU(),
                batchnorm=self.batchnorm,
                spectral_norm=self.spectral_norm,
            ),
            Conv2dScaleChannels(
                scale=1,
                kernel_size=3,
                activation=nn.ReLU(),
                batchnorm=self.batchnorm,
                spectral_norm=self.spectral_norm,
            ),
        )

        self.unet_layers = []
        # We'll use a little different logic for the last 2 ones
        assert self.encoder.num_residuals >= 2  # This includes the input
        for i in range(self.encoder.num_residuals - 2):
            # We use self attention on the 3rd to last block (like in SAGAN paper)
            self_attention = i == self.encoder.num_residuals - 3
            unet_block = UnetBlockWide(
                self.channels,
                self_attention=self_attention,
                spectral_norm=self.spectral_norm,
            )
            self.unet_layers.append(unet_block)

        # And the penultimate one
        unet_block = UnetBlockWide(
            self.channels // 2, self_attention=False, spectral_norm=self.spectral_norm
        )
        self.unet_layers.append(unet_block)
        self.unet_layers = nn.ModuleList(self.unet_layers)

        # And the final one
        self.upsampler = PixelShuffle_ICNR(scale=2, spectral_norm=self.spectral_norm)
        self.merge = concat
        self.res_block = ResidualBlock(
            kernel_size=3,
            activation=nn.ReLU(),
            batchnorm=self.batchnorm,
            dense=False,
            spectral_norm=self.spectral_norm,
        )
        # Final Conv layer
        self.final_conv = Conv2d(self.input_channels, kernel_size=1)

    def forward(self, x):
        inputs = x
        # Input batchnorm if its activated
        x = self.batchnorm_in(x)
        # Encoder - Uncomment below if running out of memory and not training resnet
        # with torch.no_grad():
        #     x, residuals = self.encoder(x)
        # residuals = [r.detach() for r in residuals]
        with torch.no_grad():
            x, *residuals = self.encoder(x)
        residuals = [r.detach() for r in residuals]
        # Neck
        x = self.neck(x)
        # Unets - save the last residual for the last cross
        for residual_input, unet_layer in zip(
            reversed(residuals[1:]), self.unet_layers
        ):
            residual_input = residual_input
            x = unet_layer(x, residual_input)
        # last cross
        x = self.upsampler(x)
        x = self.merge([x, inputs, residuals[0]])
        x = self.res_block(x)
        return self.final_conv(x)
