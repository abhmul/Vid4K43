import logging
import torch
import torch.nn as nn


def Input(*input_shape):
    # Use 1 for the batch size
    return torch.zeros(1, *input_shape)


class Layer(nn.Module):
    def __init__(self):
        """An abstract class for a layer. This subclasses `torch`'s `nn.Module`.

        Implement this to get a PyJet specific layer. You can include a builder
        for the function if it cannot be built upon instantiation. Simply write
        a function that modifies your layer to have all necessary parameters
        instantiated and annotate it with the `@builder` decorator from Pyjet's
        `layers` package. Then you can call it at the top of your `forward` and
        it will only run once.
        """
        super(Layer, self).__init__()
        self.built = False
        self.build_hook = None

    def register_builder(self, builder_func, *args, **kwargs):
        def builder_wrapper(self, inputs):
            if not self.built:
                assert self.build_hook is not None, "Builder has not been registered!!"
                logging.info(f"Attempting to build {self.__class__.__name__} layer")
                builder_func(*inputs, *args, **kwargs)
                self.built = True
                # Remove the hook so it's not called again once the model is built
                logging.info("Built and removing hook for layer %r" % self)
                self.build_hook.remove()

        self.build_hook = self.register_forward_pre_hook(builder_wrapper)
        return builder_wrapper

    def forward(self, *input):
        """Implement this to define the input to output logic of the layer."""
        raise NotImplementedError()
