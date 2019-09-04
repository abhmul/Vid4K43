from layers import DynamicUnetWide
from pyjet.models import SLModel


class Generator(DynamicUnetWide, SLModel):
    def __init__(
        self, encoder, channels_factor=1, batchnorm=False, spectral_norm=False
    ):
        SLModel.__init__(self)
        DynamicUnetWide.__init__(
            self, encoder, channels_factor, batchnorm, spectral_norm
        )

    def forward(self, x):
        return DynamicUnetWide.forward(self, x)

