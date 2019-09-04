from glob import glob
import argparse

import numpy as np
from kaggleutils import set_random_seed, get_random_seed
from pyjet.data import ImageDataset

from resnet import *
from config import TRAINING_CONFIG as TC
from data import RandomCropper, CrappifyTransformer, ImageNetNormalizer
from loss import FeatureLoss
from model import Generator


parser = argparse.ArgumentParser(description="Pretrain the 4kifier with real images")
parser.add_argument("-d", "--data", help="Glob for image data.")

set_random_seed(TC["seed"])


# TODO: Optimizer TTUR
# TODO: Set up data to be loaded channels first somewhere in the pipeline
# TODO: Write tests for basic tests for loss, optimizer, data
# TODO: Finish and test training script


def build_data_loader(dataset, batch_size, epoch_size, crop_size):
    # Rounds the epoch size up to the nearest batch
    epoch_size = ((epoch_size + batch_size - 1) // batch_size) * batch_size
    steps_per_epoch = epoch_size // batch_size

    datagen = dataset.flow(steps_per_epoch, batch_size, shuffle=True)
    datagen = RandomCropper(crop_size)(datagen)
    datagen = CrappifyTransformer()(datagen)
    datagen = ImageNetNormalizer()(datagen)
    return datagen


def prepare_data(
    data_glob,
    batch_size=TC["batch_size"],
    epoch_size=TC["epoch_size"],
    crop_size=TC["crop_size"],
):
    img_paths = np.array(glob(data_glob))
    dataset = ImageDataset(img_paths)
    train_dataset, test_dataset = dataset.validation_split(
        split=0.1, shuffle=True, seed=get_random_seed()
    )
    train_dataset, val_dataset = train_dataset.validation_split(
        split=0.1, shuffle=True, seed=get_random_seed()
    )

    # Construct the loader
    train_datagen = build_data_loader(train_dataset, batch_size, epoch_size, crop_size)
    val_datagen = build_data_loader(val_dataset, batch_size, epoch_size, crop_size))
    test_datagen = build_data_loader(test_dataset, batch_size, epoch_size, crop_size))

    return train_datagen, val_datagen, test_datagen
