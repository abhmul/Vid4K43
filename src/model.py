import torch
import pyjet.backend as J
from pyjet.models import SLModel
from pyjet.layers import Conv2D, Input

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
        input_shape,
        encoder,
        channels_factor=1,
        batchnorm=False,
        spectral_norm=False,
        input_batchnorm=False,
        epsilon=J.epsilon,
    ):
        SLModel.__init__(self)
        encoder = RESNET_DICT[encoder](pretrained=True)
        DynamicUnetWide.__init__(
            self, encoder, channels_factor, batchnorm, spectral_norm, input_batchnorm
        )
        self.imagenet_mean = J.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = J.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.epsilon = epsilon

        print(f"Inferring with input shape {input_shape}")
        self.infer_inputs(Input(*input_shape))

    def imagenet_normalize(self, x):
        return (x - self.imagenet_mean) / (self.imagenet_std)

    def imagenet_unnormalize(self, x):
        return x * self.imagenet_std + self.imagenet_mean

    def forward(self, x):
        # x = self.imagenet_normalize(x)
        # Comes out in range (-inf, inf)
        x = DynamicUnetWide.forward(self, x)
        x = torch.sigmoid(x)
        self.loss_in = x
        # The loss has a VGG component, so we imagenet-normalize
        # self.loss_in = self.imagenet_normalize(x)
        return x


class GeneratorTest0(SLModel):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__()
        print("Constructing TEST GEN 0")
        # No params for this, use one to satisfy optimizer
        self.param = torch.nn.Parameter(J.tensor(0.0))

        print(f"Inferring with input shape {input_shape}")
        self.infer_inputs(Input(*input_shape))

    def trainable_params(self):
        return set(self.parameters())

    def forward(self, x):
        assert x.max() <= 1.0
        assert x.min() >= 0.0
        # Just return the input
        self.loss_in = x + self.param - self.param
        return x


class GeneratorTest1(SLModel):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__()
        print("Constructing TEST GEN 1")
        # Use 3 conv layers on the input then add the input
        self.conv1 = Conv2D(16, 5, activation="relu", batchnorm=True)
        self.conv2 = Conv2D(32, 5, activation="relu", batchnorm=True)
        self.conv3 = Conv2D(3, 5, activation="sigmoid", batchnorm=True)

        print(f"Inferring with input shape {input_shape}")
        self.infer_inputs(Input(*input_shape))

    def trainable_params(self):
        return set(self.parameters())

    def forward(self, x):
        assert x.max() <= 1.0
        assert x.min() >= 0.0
        inputs = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self.loss_in = (x + inputs) / 2
        return self.loss_in
