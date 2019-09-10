import torch
import pyjet.backend as J
from pyjet.models import SLModel

from layers import DynamicUnetWide
from resnet import *

RESNET_DICT = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class Generator(DynamicUnetWide, SLModel):
    def __init__(
        self,
        encoder,
        channels_factor=1,
        batchnorm=False,
        spectral_norm=False,
        epsilon=J.epsilon,
    ):
        SLModel.__init__(self)
        encoder = RESNET_DICT[encoder](pretrained=True)
        DynamicUnetWide.__init__(
            self, encoder, channels_factor, batchnorm, spectral_norm
        )
        self.imagenet_mean = J.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = J.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.epsilon = epsilon

    def uint8_to_float(self, x):
        return x.float() / 255.0

    def float_to_uint8(self, x):
        return (x * 255.0).byte()

    def imagenet_normalize(self, x):
        return (x - self.imagenet_mean) / (self.imagenet_std)

    def imagenet_unnormalize(self, x):
        return x * self.imagenet_std + self.imagenet_mean

    def forward(self, x):
        x = self.uint8_to_float(x)
        x = self.imagenet_normalize(x)
        # Comes out in range (-inf, inf)
        x = DynamicUnetWide.forward(self, x)
        x = torch.sigmoid(x)
        # The loss has a VGG component, so we imagenet-normalize
        self.loss_in = self.imagenet_normalize(x)
        # The output is gonna be a uint8 array
        return self.float_to_uint8(x)

