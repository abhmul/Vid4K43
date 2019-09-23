import torch.nn as nn
import torch.nn.functional as F
from loss import accuracy_with_logits

from abstract_model import Model
from layers import Critic


class CriticPretrain(Model, Critic):
    def train_step(self, batch):
        x, y = batch["input"], batch["target"]
        pred = self(x)
        y = torch.full_like(pred, y)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = accuracy_with_logits(pred, y)

        return {"loss": loss, "acc": acc}

    def val_step(self, batch):
        score_dict = self.train_step(batch)
        return {"val_" + key: score for key, score in score_dict.items()}

