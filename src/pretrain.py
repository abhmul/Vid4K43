from glob import glob
import argparse

import numpy as np
from torch.optim import SGD
from kaggleutils import set_random_seed, get_random_seed
from pyjet.data import ImageDataset
from pyjet.callbacks import OneCycleScheduler, Plotter, ModelCheckpoint

from config import TRAINING_CONFIG as TC
from data import RandomCropper, CrappifyTransformer, ChannelsFirstTransformer
from loss import FeatureLoss
from model import Generator


parser = argparse.ArgumentParser(description="Pretrain the 4kifier with real images")
parser.add_argument("-d", "--data", help="Glob for image data.")

set_random_seed(TC["seed"])


# TODO: Finish evaluation script
# TODO: Add logging


def build_data_loader(dataset, batch_size, epoch_size, crop_size):
    # Rounds the epoch size up to the nearest batch
    steps_per_epoch = (epoch_size + batch_size - 1) // batch_size

    datagen = dataset.flow(steps_per_epoch, batch_size, shuffle=True)
    datagen = RandomCropper(crop_size)(datagen)
    datagen = CrappifyTransformer()(datagen)
    datagen = ChannelsFirstTransformer()(datagen)
    return datagen


def prepare_data(
    data_glob,
    batch_size=TC["batch_size"],
    epoch_size=TC["epoch_size"],
    crop_size=TC["crop_size"],
    test_split=TC["test_split"],
):
    # Get splits
    train_split = 1 - test_split
    val_split = train_split * test_split  # Use this for getting the val epoch size

    img_paths = np.array(glob(data_glob))
    dataset = ImageDataset(img_paths, to_float=False)
    train_dataset, test_dataset = dataset.validation_split(
        split=test_split, shuffle=True, seed=get_random_seed()
    )
    train_dataset, val_dataset = train_dataset.validation_split(
        split=test_split, shuffle=True, seed=get_random_seed()
    )

    # Construct the loader
    train_datagen = build_data_loader(train_dataset, batch_size, epoch_size, crop_size)
    val_datagen = build_data_loader(
        val_dataset, batch_size, epoch_size * val_split, crop_size
    )
    test_datagen = build_data_loader(
        test_dataset, batch_size, epoch_size * test_split, crop_size
    )

    return train_datagen, val_datagen, test_datagen


def prepare_model(
    # Model Params
    encoder=TC["encoder"],
    channels_factor=TC["channels_factor"],
    batchnorm=TC["batchnorm"],
    spectral_norm=TC["spectral_norm"],
    # Loss Params
    layer_weights=TC["layer_weights"],
    # Optimizer Params
    lr_range=TC["lr_range"],
    momentum_range=TC["momentum_range"],
    period=TC["period"],
):
    # Model
    model = Generator(encoder, channels_factor, batchnorm, spectral_norm)
    # Loss
    feature_loss = FeatureLoss(layer_weights)
    model.add_loss_with_aux(
        feature_loss,
        inputs="loss_in",
        auxilaries=feature_loss.auxilaries,
        name="feat_loss",
    )
    # Optimizer
    lr = lr_range[0]
    momentum = momentum_range[0]
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    model.add_optimizer(optimizer, name="sgd")
    # Bind to the model to package them together
    model.lr_scheduler = OneCycleScheduler(optimizer, lr_range, momentum_range, period)

    return model


def pretrain(train_datagen, val_datagen, epochs=TC["epochs"]):
    model = prepare_model()

    # other callbacks
    # This will save the best scoring model weights to the current directory
    best_model = ModelCheckpoint(
        "gen_pretrain" + ".state",
        monitor="val_feat_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    # This will plot the model's accuracy during training
    plotter = Plotter(scale="log", monitor="feat_loss")

    model.fit_generator(
        train_datagen,
        train_datagen.steps_per_epoch,
        epochs,
        val_datagen,
        val_datagen.steps_per_epoch,
        callbacks=[model.lr_scheduler, best_model, plotter],
        verbose=1,
    )

    return model


def evaluate(test_datagen, model: Generator):
    """An evaluation loop that runs each batch in the test gen and displays the original - prediction - GT"""
    NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args()
    train_datagen, val_datagen, test_datagen = prepare_data(args.data_glob)
    model = prepare_model()

    model = pretrain(train_datagen, val_datagen)
