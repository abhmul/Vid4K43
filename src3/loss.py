import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import cfgs as VGG_CONFIGS

from pyjet.hooks import hook_outputs


def requires_grad(module, grad):
    parameters = list(module.parameters())
    assert parameters, "Module has no parameters!"
    for param in parameters:
        param.requires_grad = grad


def get_feature_layers():
    vgg16_config = VGG_CONFIGS["D"]


class FeatureLoss(nn.Module):
    def __init__(self, layer_weights=[20, 70, 10]):
        super().__init__()
        self.layer_weights = layer_weights

        # Get the features and turn off their grad
        self.features = models.vgg16_bn(pretrained=True).features.eval()
        requires_grad(self.features, False)

        # Get the layers we want
        pre_max_pool_inds = [
            i - 1
            for i, layer in enumerate(self.features.children())
            if isinstance(layer, nn.MaxPool2d)
        ]
        layer_ids = pre_max_pool_inds[2:5]  # We select the middle 3

        # Hook the features to use for the loss
        self.loss_features = [self.features[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.base_loss = F.l1_loss

    def _make_features(self, x):
        self.features(x)
        return [o[0] for o in self.hooks.stored]

    def forward(self, x, y):
        out_feat = self._make_features(y)
        assert len(out_feat) == 3, f"Incourrect output features {len(out_feat)}"
        in_feat = self._make_features(x)
        pixel = self.base_loss(x, y)
        feat1 = self.base_loss(in_feat[0], out_feat[0]) * self.layer_weights[0]
        feat2 = self.base_loss(in_feat[1], out_feat[1]) * self.layer_weights[1]
        feat3 = self.base_loss(in_feat[2], out_feat[2]) * self.layer_weights[2]

        loss = pixel + feat1 + feat2 + feat3
        return {
            "loss": loss,
            "pixel": pixel,
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
        }


# Hack to not save weights in model state dict
__feature_loss = FeatureLoss()


def feature_loss(x, y):
    return __feature_loss(x, y)


def accuracy_with_logits(logits, y):
    preds = torch.round(torch.sigmoid(logits))
    acc = torch.mean((preds == y).float())
    return acc.item()

