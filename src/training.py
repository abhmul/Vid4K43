from torch.optim import SGD
from pyjet.metrics import Metric

from data import PrepareData
from model import Generator, GeneratorTest0, GeneratorTest1
from loss import FeatureLoss


class LR(Metric):
    def __init__(self, onecycle):
        super().__init__()
        self.onecycle = onecycle

    def __call__(self, y_pred, y_true):
        return self.score(y_pred, y_true)

    def score(self, y_pred, y_true):
        return J.tensor(self.onecycle.lr)

    def accumulate(self):
        return self.onecycle.lr

    def reset(self):
        return self


class Momentum(Metric):
    def __init__(self, onecycle):
        super().__init__()
        self.onecycle = onecycle

    def __call__(self, y_pred, y_true):
        return self.score(y_pred, y_true)

    def score(self, y_pred, y_true):
        return J.tensor(self.onecycle.momentum)

    def accumulate(self):
        return self.onecycle.momentum

    def reset(self):
        return self


def build_data_loader(
    dataset, batch_size, epoch_size, crop_size, resize_factor, noise, shuffle=True
):
    # Rounds the epoch size up to the nearest batch
    steps_per_epoch = (epoch_size + batch_size - 1) // batch_size

    print(f"[build_data_loader] Using batch size {batch_size}")
    print(f"[build_data_loader] Using {steps_per_epoch} steps per epoch")
    datagen = dataset.flow(steps_per_epoch, batch_size, shuffle=shuffle)
    datagen = PrepareData(crop_size, resize_factor, noise=noise)(datagen)
    return datagen


def build_generator(
    input_shape,
    encoder,
    channels_factor,
    batchnorm,
    spectral_norm,
    input_batchnorm,
    load_model="",
    test=-1,
):
    # Figure out which generator to use (eg when we test)
    gen_class = Generator
    if test == 0:
        print("[build_generator] Using GeneratorTest0")
        gen_class = GeneratorTest0
    elif test == 1:
        print("[build_generator] Using GeneratorTest1")
        gen_class = GeneratorTest1

    on = lambda is_on: "on" if is_on else "off"
    print(
        f"[build_generator] Building model with encoder {encoder}, "
        f"channels_factor {channels_factor}, "
        f"batchnorm {on(batchnorm)}, "
        f"spectral_norm {on(spectral_norm)}"
        f"input_batchnorm {on(input_batchnorm)}"
    )
    model = gen_class(
        input_shape,
        encoder,
        channels_factor=channels_factor,
        batchnorm=batchnorm,
        spectral_norm=spectral_norm,
        input_batchnorm=input_batchnorm,
    )

    if load_model:
        print(f"[build_generator] Loading generator weights from {load_model}")
        model.load_state(load_model)

    return model


def setup_gen_train(model, layer_weights, lr=1e-5, momentum=0.95):
    print(f"[setup_gen_train] Using weights {layer_weights} for loss.")
    feature_loss = FeatureLoss(layer_weights)
    model.add_loss_with_aux(
        feature_loss, inputs="loss_in", auxilaries=feature_loss.auxilaries, name="pixel"
    )
    optimizer = SGD(model.trainable_params(), lr=lr, momentum=momentum)
    model.add_optimizer(optimizer, name="sgd")

    return model

