import os
import argparse
import torch
import numpy as np

from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models import GeneratorPretrain

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print("loading model...")
    model = GeneratorPretrain(hparams)
    print("model built")

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------

    # init experiment
    exp = Experiment(
        name=hparams.experiment_name,
        save_dir=hparams.test_tube_save_path,
        autosave=True,
        description="run1",
    )

    exp.argparse(hparams)
    exp.save()

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    model_save_path = "{}/{}/{}".format(hparams.model_save_path, exp.name, exp.version)

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        experiment=exp, checkpoint_callback=checkpoint, gpus=hparams.gpus, use_amp=True
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    # dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(root_dir, "output")
    checkpoint_dir = os.path.join(output_dir, "weights")
    test_tube_dir = os.path.join(output_dir, "test_tube_data")

    # although we user hyperOptParser, we are using it only as argparse right now
    parent_parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)

    # gpu args
    parent_parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help="how many gpus to use in the node."
        "value -1 uses all the gpus on the node",
    )
    parent_parser.add_argument(
        "--test_tube_save_path",
        type=str,
        default=test_tube_dir,
        help="where to save logs",
    )
    parent_parser.add_argument(
        "--model_save_path",
        type=str,
        default=checkpoint_dir,
        help="where to save model",
    )
    parent_parser.add_argument(
        "--experiment_name", type=str, default="exp1", help="test tube exp name"
    )

    parser = GeneratorPretrain.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print(f"RUNNING INTERACTIVE MODE ON GPUS. gpu ids: {hyperparams.gpus}")
    main(hyperparams)

