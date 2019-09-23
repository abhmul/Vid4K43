import torch.nn as nn
from loss import feature_loss

from abstract_model import Model
from layers import DynamicUnetWide
from loss import feature_loss


class GeneratorPretrain(Model, DynamicUnetWide):
    def train_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]
        gen_x = self(crap_x)
        loss_dict = feature_loss(gen_x, good_x)
        return loss_dict

    def val_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]
        gen_x = self(crap_x)
        loss_dict = feature_loss(gen_x, good_x)
        return {"val_" + key: loss for key, loss in loss_dict.items()}

