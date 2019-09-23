import torch.nn as nn

from . import Layer
from . import utils


class BatchNorm2d(Layer):
    def __init__(self, input_shape=None):
        """Pyjet's implementation of an input-inferring BatchNormalization layer"""
        super().__init__()
        self.input_shape = input_shape

        self.bn = None
        self.in_channels = None

        # Registrations
        self.register_builder(self.__build_layer)

    def __build_layer(self, inputs):
        self.input_shape = utils.get_sample_shape(inputs)
        self.in_channels = utils.get_num_channels(inputs)
        self.bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, inputs):
        return self.bn(inputs)
