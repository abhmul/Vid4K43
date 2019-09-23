import torch
import torch.nn as nn

from . import Layer, Conv2d, SelfAttention
from .utils import flatten


class Critic(Layer):
    def __init__(self, input_channels=3, filters=256, blocks=3, dropout=0.15):
        self.input_channels = input_channels
        self.filters = filters
        self.blocks = blocks
        self.dropout = dropout

        self.leaky = nn.LeakyReLU(0.2)
        self.conv_in = Conv2d(
            filters,
            kernel_size=4,
            stride=2,
            activation=self.leaky,
            dropout=self.dropout / 2,
            spectral_norm=True,
        )

        self.conv_middle = []
        for i in range(self.blocks):
            conv1 = Conv2d(
                filters,
                kernel_size=3,
                activation=self.leaky,
                dropout=self.dropout,
                spectral_norm=True,
            )
            conv2 = Conv2d(
                filters * 2,
                kernel_size=4,
                stride=2,
                activation=self.leaky,
                spectral_norm=True,
            )
            self_attention = SelfAttention()
            self.conv_middle += [conv1, conv2, self_attention]
            filters *= 2
        self.conv_middle = nn.ModuleList(self.conv_middle)

        self.conv_out = nn.ModuleList(
            [
                Conv2d(
                    filters, kernel_size=3, activation=self.leaky, spectral_norm=True
                ),
                Conv2d(1, kernel_size=4, bias=False, padding=0, spectral_norm=True),
            ]
        )

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.conv_middle:
            x = layer(x)
        for layer in self.conv_out:
            x = layer(x)
        return flatten(x)

