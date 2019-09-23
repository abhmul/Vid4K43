import torch.nn as nn


class Model(nn.Module):
    # Hack to avoid overriding init
    def __new__(cls):
        module = super().__new__(cls)
        module.__optimizers = []
        module.__schedulers = []
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
        self.train_end(outputs)

    def predict_step(self, batch):
        # Overrideable
        return self(batch["input"])

    def add_optimizer(self, optim, name):
        # DON'T OVERRIDE
        setattr(self, name, optim)
        self.__optimizers.append(optim)

    def add_schedule(self, schedule, name):
        # DON'T OVERRIDE
        setattr(self, name, scheduler)
        self.__schedulers.append(schedule)

    def optimizers(self):
        # Overrideable
        if not self.__optimizers:
            raise NotImplementedError()
        return self.__optimizers

    def schedulers(self):
        # Overrideable
        return self.__schedulers

    def save(self, path):
        torch.save(self.state_dict, path)
