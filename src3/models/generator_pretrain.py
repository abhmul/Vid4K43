import torch.nn as nn

from .abstract_model import Model
from layers import DynamicUnetWide
from loss import FeatureLoss


class GeneratorPretrain(Model, DynamicUnetWide):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_loss = FeatureLoss()

    def train_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]
        gen_x = self(crap_x)
        loss_dict = self.feature_loss(gen_x, good_x)
        return loss_dict

    def val_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]
        gen_x = self(crap_x)
        loss_dict = self.feature_loss(gen_x, good_x)
        return {"val_" + key: loss for key, loss in loss_dict.items()}

    def state_dict(self):
        d = super().state_dict()
        loss_d = {
            "feature_loss." + k: v for k, v in self.feature_loss.state_dict().items()
        }
        new_d = {k: v for k, v in d.items() if k not in loss_d}
        return new_d

