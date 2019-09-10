import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import cfgs as VGG_CONFIGS

from pyjet.hooks import hook_outputs
import pyjet.backend as J


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
        if J.use_cuda:
            self.features = self.features.cuda()
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

        self.pixel, self.feat1, self.feat2, self.feat3 = [None] * 4
        self.auxilaries = ["pixel", "feat1", "feat2", "feat3"]

    def _make_features(self, x):
        self.features(x)
        return list(self.hooks.stored)

    def forward(self, x, y):
        out_feat = self._make_features(y)
        assert len(out_feat) == 3, f"Incourrect output features {len(out_feat)}"
        in_feat = self._make_features(x)
        self.pixel = self.base_loss(x, y)
        self.feat1 = self.base_loss(in_feat[0], out_feat[0], self.wgts[0])
        self.feat2 = self.base_loss(in_feat[1], out_feat[1], self.wgts[1])
        self.feat3 = self.base_loss(in_feat[2], out_feat[2], self.wgts[2])

        return sum(self.feat_losses)

