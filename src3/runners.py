import os
from collections import defaultdict
import itertools as itools
from operator import le, ge

from tqdm import tqdm

import torch
import torch.nn as nn


def get_current_device():
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")


def batch_to_device(batch, device):
    return {key: tensor.to(device) for key, tensor in batch.items()}


def none_func(*args, **kwargs):
    pass


def outputs_as_item(output_dict):
    return {k: t.item() for k, t in output_dict.items()}


class ModelRunner(object):
    def __init__(self, steps=None, device=get_current_device(), progress=False):
        """If steps=-1, run data loader to completeion"""
        self.device = torch.device(device)
        self.steps = steps
        self.progress = progress

        # Used by any runner that needs to save outputs
        self.outputs = None

        print(f"{self.__class__.__name__} running on device {self.device}")

    def setup_model(self, model):
        print(f"Sending model to device {self.device}")
        return model.to(self.device)

    def setup_dataloader(self, dataloader):
        steps = len(dataloader)
        if self.steps is not None:
            steps = self.steps
            dataloader = itertools.islice(dataloader, steps)
        dataloader = tqdm(dataloader, total=steps, disable=not self.progress)
        return dataloader


class Predictor(ModelRunner):
    @torch.no_grad()
    def __call__(self, model, dataloader):
        model = self.setup_model(model).eval()
        dataloader = self.setup_dataloader(dataloader)
        for i, batch in enumerate(dataloader):
            # Assumes batch is a dict of string -> tensors
            batch = batch_to_device(batch, self.device)
            yield i, model.predict_step(batch)


class Evaluator(ModelRunner):
    @torch.no_grad()
    def __call__(self, model, dataloader):
        model = self.setup_model(model).eval()
        dataloader = self.setup_dataloader(dataloader)

        self.outputs = defaultdict(list)
        for i, batch in enumerate(dataloader):
            batch = batch_to_device(batch, self.device)
            output = model.val_step(batch)
            # If it's not implemented we're done
            if output is NotImplemented:
                return

            output = outputs_as_item(output)
            dataloader.set_postfix(output)
            # Add the output to the outputs tracker
            for key, val in output.items():
                self.outputs[key].append(val)

            yield i, output

        self.outputs = model.val_end(self.outputs)
        dataloader.set_postfix(self.outputs)


class EpochTrainer(ModelRunner):
    def __call__(self, model, dataloader):
        model = self.setup_model(model).train()
        dataloader = self.setup_dataloader(dataloader)

        self.outputs = defaultdict(list)
        for i, batch in enumerate(dataloader):
            batch = batch_to_device(batch, self.device)
            output = model.train_step(batch)

            # Run the training routine
            [optim.zero_grad() for optim in model.optimizers()]
            loss = output["loss"]
            loss.backward()
            [optim.step() for optim in model.optimizers()]
            [scheduler.step() for scheduler in model.schedulers()]

            output = outputs_as_item(output)
            dataloader.set_postfix(output)
            # Add the output to the outputs tracker
            for key, val in output.items():
                self.outputs[key].append(val)

            yield i, output

        self.outputs = model.train_end(self.outputs)
        dataloader.set_postfix(self.outputs)
        model.eval()


class Checkpointer(object):
    def __init__(self, path, save_best=False, track="val_loss", method=le):
        self.path = path
        self.save_best = save_best
        self.track = track
        self.method = method

        self.track_val = None

    def add_format_kwargs(self, **format_kwargs):
        rest, ext = os.path.splitext(self.path)
        return rest + "_{epoch}".format(**format_kwargs) + ext

    def __call__(self, model, outputs, **format_kwargs):
        if not self.save_best:
            model.save(self.add_format_kwargs(**format_kwargs))

        track_val = outputs[self.track]
        if self.track_val is None or self.method(track_val, self.track_val):
            print(
                f"Model improved tracked {self.track} from {self.track_val} to {track_val}"
            )
            model.save(self.path)
            self.track_val = track_val


class Trainer(ModelRunner):
    def __init__(
        self,
        epochs,
        steps_per_epoch=None,
        validation_steps=None,
        device=get_current_device(),
        progress=True,
    ):
        super().__init__(steps=steps_per_epoch, device=device, progress=progress)
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        self.epoch_trainer = EpochTrainer(
            steps=steps_per_epoch, device=device, progress=progress
        )
        # This will immediately return a stopIteration if there is no model.val_step(...) defined
        self.evaluator = Evaluator(
            steps=validation_steps, device=device, progress=progress
        )

        # General trackers
        self.global_step = 0
        self.validation_global_step = 0

    def __call__(
        self,
        model,
        train_dataloader,
        validation_dataloader=None,
        writer=None,
        train_tag="train",
        validation_tag="validation",
    ):
        self.outputs = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            for i, output in self.epoch_trainer(model, train_dataloader):
                self.global_step += 1
                if writer is not None:
                    writer.add_scalars(train_tag, output, self.global_step)
            train_outputs = self.epoch_trainer.outputs

            for i, output in self.evaluator(model, validation_dataloader):
                self.validation_global_step += 1
                if writer is not None:
                    writer.add_scalars(validation_tag, output, self.global_step)
            # Even if it doesn't run, it still defines an outputs
            val_outputs = self.evaluator.outputs
            assert (
                val_outputs is not None
            ), "Evaluator should define an outputs upon being called, currently None."

            output = {**train_outputs, **val_outputs}
            self.outputs.append(output)
            yield i, output

