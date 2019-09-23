from glob import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD
from kaggleutils import set_random_seed, get_random_seed
from pyjet.data import ImageDataset
from pyjet.callbacks import OneCycleScheduler, Plotter, ModelCheckpoint
from pyjet.metrics import Metric
import pyjet.backend as J

from config import TRAINING_CONFIG as TC
from training import build_data_loader, build_generator, setup_gen_train, LR, Momentum


parser = argparse.ArgumentParser(description="Pretrain the 4kifier with real images")
parser.add_argument(
    "-d",
    "--data",
    default="/media/abhmul/BackupSSD1/Datasets/open-image/images/*",
    help="Glob for image data.",
)
parser.add_argument(
    "-tr",
    "--train",
    action="store_true",
    default=False,
    help="Run the script to train a model",
)
parser.add_argument(
    "-te",
    "--test",
    action="store_true",
    default=False,
    help="Run the script to test a model",
)
parser.add_argument("--model_path", help="Path to trained model to load")


set_random_seed(TC["seed"])


def prepare_data(
    data_glob,
    # Config Variables
    batch_size,
    epoch_size,
    crop_size,
    resize_factor,
    noise,
    test_split,
    test_batch_size,
    test_crop_size,
    # Unused
    **kwargs,
):
    print(f"Matching images with glob {data_glob}")
    img_paths = np.array(glob(data_glob))
    dataset = ImageDataset(img_paths, to_float=False)

    print(f"[prepare_data] Using splits test({test_split})")
    train_dataset, test_dataset = dataset.validation_split(
        split=test_split, shuffle=True, seed=get_random_seed()
    )
    train_dataset, val_dataset = train_dataset.validation_split(
        split=test_split, shuffle=True, seed=get_random_seed()
    )
    print(f"[prepare_data] Train Images: {len(train_dataset)}")
    print(f"[prepare_data] Val Images: {len(val_dataset)}")
    print(f"[prepare_data] Test Images: {len(test_dataset)}")

    # Construct the loader
    print("[prepare_data] Building train datagen")
    train_datagen = build_data_loader(
        train_dataset, batch_size, epoch_size, crop_size, resize_factor, noise
    )
    print("[prepare_data] Building val datagen")
    val_epoch_size = int(epoch_size * test_split)
    val_datagen = build_data_loader(
        val_dataset, batch_size, val_epoch_size, crop_size, resize_factor, noise
    )
    print("[prepare_data] Building test datagen")
    test_epoch_size = int(epoch_size * test_split)
    test_datagen = build_data_loader(
        test_dataset,
        test_batch_size,
        test_epoch_size,
        test_crop_size,
        resize_factor,
        noise,
    )

    return train_datagen, val_datagen, test_datagen


def prepare_model(
    # Model Params
    input_size,
    encoder,
    channels_factor,
    batchnorm,
    spectral_norm,
    input_batchnorm,
    # Loss Params
    layer_weights,
    # Unused
    **kwargs,
):
    input_shape = (3,) + input_size
    model = build_generator(
        input_shape,
        encoder,
        channels_factor,
        batchnorm,
        spectral_norm,
        input_batchnorm,
        test=-1,
    )
    setup_gen_train(model, layer_weights)

    return model


def pretrain(
    model,
    train_datagen,
    val_datagen,
    # Train param
    epochs,
    # LR Scheduler Params
    lr_range,
    momentum_range,
    period_factor,
    fname,
    # Unused
    **kwargs,
):
    period = period_factor * train_datagen.steps_per_epoch
    scheduler = OneCycleScheduler(
        model.get_optimizer("sgd"), lr_range, momentum_range, period
    )

    # other callbacks
    # This will save the best scoring model weights to the current directory
    best_model = ModelCheckpoint(
        fname + ".state", monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )
    # This will plot the model's accuracy during training
    plotter = Plotter(scale="log", monitor="loss", save_to_file=fname + "_plot.png")

    model.fit_generator(
        train_datagen,
        train_datagen.steps_per_epoch,
        epochs,
        val_datagen,
        val_datagen.steps_per_epoch,
        callbacks=[scheduler, best_model, plotter],
        metrics=[LR(scheduler), Momentum(scheduler)],
        verbose=1,
    )

    return model


def plot_output(original, enhanced, ground_truth):
    fig = plt.figure(figsize=(32, 96))

    def create_ax(loc, i):
        ax = fig.add_subplot(*loc, i + 1)
        ax.axis("off")
        ax.set_aspect("equal")

    def transpose_channels(x):
        return x.transpose(1, 2, 0)

    create_ax((1, 3), 0)
    plt.imshow(transpose_channels(original))
    create_ax((1, 3), 1)
    plt.imshow(transpose_channels(enhanced))
    create_ax((1, 3), 2)
    plt.imshow(transpose_channels(ground_truth))
    plt.show()


def evaluate(test_datagen, model, model_path):
    """An evaluation loop that runs each batch in the test gen and displays the original - prediction - GT"""
    # Load the model
    print(f"Loading model from {model_path}")
    model.load_state(model_path)
    for x, y in test_datagen:
        print(x.shape)
        print(y.shape)
        pred = model.predict_on_batch(x)
        for xi, predi, yi in zip(x, pred, y):

            plot_output(xi, predi, yi)


if __name__ == "__main__":
    args = parser.parse_args()
    tr, va, te = prepare_data(args.data, **TC)
    model = prepare_model(**TC)
    if args.train:
        model = pretrain(model, tr, va, **TC)
    if args.test:
        evaluate(te, model, args.model_path)
