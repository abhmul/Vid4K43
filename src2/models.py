from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser

from data import CrappifyDataset, Crappify
from layers import Input, DynamicUnetWide
from layers.resnet import *
from loss import FeatureLoss

RESNET_DICT = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


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


class GeneratorPretrain(pl.LightningModule, DynamicUnetWide):
    def __init__(
        self,
        # tng_dataloader,
        # input_shape,
        # encoder,
        # channels_factor=1,
        # batchnorm=False,
        # spectral_norm=False,
        # input_batchnorm=False,
        # val_dataloader=None,
        # lr_max=1e-2,
        # momentum=0.9,
        # loss_weights=[20, 70, 10],
        hparams,
    ):
        self.hparams = hparams
        encoder = RESNET_DICT[hparams.encoder](pretrained=True)
        super(GeneratorPretrain, self).__init__(
            encoder,
            hparams.channels_factor,
            hparams.batchnorm,
            hparams.spectral_norm,
            hparams.input_batchnorm,
        )

        self.batch_size = hparams.batch_size
        self.test_batch_size = hparams.test_batch_size
        self.lr_max = hparams.lr
        self.momentum = hparams.momentum

        self.loss = FeatureLoss(
            layer_weights=[
                hparams.loss_weight1,
                hparams.loss_weight2,
                hparams.loss_weight3,
            ]
        )

        input_shape = (self.batch_size, 3, hparams.crop_size, hparams.crop_size)
        print(f"Inferring with input shape {input_shape}")
        self.infer_inputs(torch.rand(*input_shape))

        self.build_datasets()
        print(
            f"Built datasets with length train: {len(self.train_dataset)} - val: {len(self.val_dataset)} - test: {len(self.test_dataset)}"
        )

    def trainable_params(self):
        params = set(self.parameters())
        loss_params = set(self.loss.parameters())
        encoder_params = set(self.encoder.parameters())
        return list(params - loss_params - encoder_params)

    def build_datasets(self):
        data = CrappifyDataset(
            self.hparams.glob_path,
            crappifier=Crappify(self.hparams.crap_factor),
            crop_size=self.hparams.crop_size,
        )
        test_num_samples = round(len(data) * self.hparams.test_split)
        self.test_dataset, self.val_dataset, self.train_dataset = random_split(
            data, [test_num_samples, test_num_samples, len(data) - 2 * test_num_samples]
        )
        self.test_dataset.crop_size = self.hparams.test_crop_size

    def infer_inputs(self, inputs):
        with torch.no_grad():
            self.forward(inputs)

    def forward(self, x):
        # TODO: Imagenet normalize?
        x = DynamicUnetWide.forward(self, x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).half()
        loss = self.loss(y_hat, y)

        output = {
            "loss": loss,
            "prog": {
                "loss": loss,
                "pixel": self.loss.pixel,
                "feat1": self.loss.feat1,
                "feat2": self.loss.feat2,
                "feat3": self.loss.feat3,
            },
        }

        return output

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).half()
        loss = self.loss(y_hat, y)

        output = {
            "val_loss": loss,
            "val_pixel": self.loss.pixel,
            "val_feat1": self.loss.feat1,
            "val_feat2": self.loss.feat2,
            "val_feat3": self.loss.feat3,
        }

        return output

    def validation_end(self, outputs):
        def mean(nums):
            return (sum(nums) / len(nums)).item()

        keys = outputs[0].keys()
        return {mean([output[key] for output in outputs]) for key in keys}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x).half()
        try:
            x = x.cpu()
            y = y.cpu()
            y_hat = y_hat.cpu()
        except:
            pass
        x = x.numpy()
        y = y.numpy()
        y_hat = y_hat.numpy()
        plot_output(x, y, y_hat)

    @pl.data_loader
    def tng_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size)

    def configure_optimizers(self):
        opt = SGD(
            self.trainable_params(),
            lr=self.lr_max,
            momentum=self.momentum,
            nesterov=True,
        )
        sched = CosineAnnealingWarmRestarts(opt, len(self.train_dataset) * 5)
        return [opt], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(
            strategy=parent_parser.strategy, parents=[parent_parser]
        )

        # network params
        parser.add_argument("--encoder", default="resnet18")
        parser.opt_list("--channels_factor", default=1, options=[1, 2, 4], type=int)
        parser.add_argument("--batchnorm", action="store_true")
        parser.add_argument("--spectral_norm", action="store_true")
        parser.add_argument("--input_batchnorm", action="store_true")
        parser.add_argument("--loss_weight1", default=20, type=int)
        parser.add_argument("--loss_weight2", default=70, type=int)
        parser.add_argument("--loss_weight3", default=10, type=int)

        # Training params
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--test_batch_size", default=1, type=int)
        parser.add_argument("--lr", default=1e-2, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--crop_size", default=256, type=int)
        parser.add_argument("--test_crop_size", default=512, type=int)
        parser.opt_list("--crap_factor", default=0.5, options=[0.25, 0.5], type=float)
        parser.add_argument("--test_split", default=0.05, type=float)

        # data
        parser.add_argument(
            "--glob_path",
            default="/media/abhmul/BackupSSD1/Datasets/open-image/images/*",
        )

        return parser

