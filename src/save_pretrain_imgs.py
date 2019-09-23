from glob import glob
import argparse

import numpy as np
from kaggleutils import safe_open_dir, set_random_seed

from pyjet.data import ImageDataset
from pyjet.models import SLModel

from config import TRAINING_CONFIG as TC
from training import build_data_loader, build_generator


parser = argparse.ArgumentParser(
    description="Save the output of the pretrained 4kifier"
)
parser.add_argument("n", type=int, help="Number of images to save")
parser.add_argument(
    "-d",
    "--data",
    default="/media/abhmul/BackupSSD1/Datasets/open-image/images/*",
    help="Glob for image data.",
)
parser.add_argument(
    "--model_path",
    default="gen_pretrain_05.state",
    help="Path to trained model to load",
)

set_random_seed(TC["seed"])


def prepare_data(
    data_glob,
    num_imgs,
    test_batch_size,
    test_crop_size,
    resize_factor,
    noise,
    # Unused
    **kwargs,
):
    print(f"Matching images with glob {data_glob}")
    img_paths = np.array(glob(data_glob))
    dataset = ImageDataset(img_paths, to_float=False)

    print("[prepare_data] Building test datagen")
    test_datagen = build_data_loader(
        dataset, test_batch_size, num_imgs, test_crop_size, resize_factor, noise, shuffle=False
    )
    test_datagen

    return test_datagen


def prepare_model(
    # Model params
    input_size,
    encoder,
    channels_factor,
    batchnorm,
    spectral_norm,
    input_batchnorm,
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

    return model


def generate_images(model: SLModel, datagen):
    preds = model.predict_generator(datagen, datagen.steps_per_epoch, verbose=1)
    for 

    



if __name__ == "__main__":
    args = parser.parse_args()
    data = prepare_data(args.data, args.n, **TC)
    model = prepare_model(**TC)

