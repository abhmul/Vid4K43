from collections import OrderedDict

import torch
import torch.nn as nn


def Input(*input_shape):
    # Use 1 for the batch size
    return torch.zeros(1, *input_shape)


class Model(nn.Module):
    # Hack to avoid overriding init
    def __new__(cls, *args, **kwargs):
        module = super().__new__(cls)
        module.__optimizers = OrderedDict()
        module.__schedulers = OrderedDict()
        return module

    def train_step(self, batch):
        # Required
        raise NotImplementedError()

    def train_end(self, outputs):
        # Overrideable
        return {
            key: torch.mean(torch.tensor(output).float()).item()
            for key, output in outputs.items()
        }

    def val_step(self, batch):
        # Optional
        return NotImplemented

    def val_end(self, outputs):
        # Overrideable
        return self.train_end(outputs)

    def predict_step(self, batch):
        # Overrideable
        return self(batch["input"])

    def add_optimizer(self, optim, name):
        # DON'T OVERRIDE
        if name in self.__optimizers:
            print(f"Overwriting old {name} {self.__optimizers[name]} with {optim}")
        setattr(self, name, optim)
        self.__optimizers[name] = optim

    def add_scheduler(self, scheduler, name):
        # DON'T OVERRIDE
        if name in self.__schedulers:
            print(f"Overwriting old {name} {self.__schedulers[name]} with {scheduler}")
        setattr(self, name, scheduler)
        self.__schedulers[name] = scheduler

    def optimizers(self):
        # Overrideable
        if not self.__optimizers:
            raise NotImplementedError("Add an optimizer prior to training the model")
        return self.__optimizers.values()

    def schedulers(self):
        # Overrideable
        return self.__schedulers.values()

    def save(self, path):
        torch.save(self.state_dict(), path)
