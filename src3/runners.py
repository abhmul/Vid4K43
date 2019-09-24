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


class ModelRunner(object):
    def __init__(self, steps=None, device=get_current_device(), progress=False):
        """If steps=-1, run data loader to completeion"""
        self.device = torch.device(device)
        self.steps = steps
        self.progress = progress

        self.steps_run = 0

        # Used by any runner that needs to save outputs
        self.outputs = None

    def setup_model(self, model):
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
            dataloader.set_postfix(output)

            # Run the training routine
            [optim.zero_grad() for optim in model.optimizers]
            loss = output["loss"]
            loss.backward()
            [optim.step() for optim in model.optimizers]
            [scheduler.step() for scheduler in model.schedulers]
            
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

        self.tracked = None

    def add_format_kwargs(self, **format_kwargs):
        rest, ext = os.path.splitext(self.path)
        return rest + "_{epoch}".format(**format_kwargs) + ext

    def __call__(self, model, outputs, **format_kwargs):
        if not save_best:
            model.save(self.add_format_kwargs(**format_kwargs))

        track_val = outputs[self.track]
        if self.tracked is None or self.method(track_val, self.tracked):
            model.save(self.path)
            self.tracked = tracked


class Trainer(object):
    def __init__(
        self,
        epochs,
        steps_per_epoch=None,
        val_steps=None,
        device=get_current_device(),
        progress=True
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        self.device = device
        self.progress = progress

        self.epoch_trainer = EpochTrainer(
            steps=steps_per_epoch, device=device, progress=progress
        )
        # This will immediately return a stopIteration if there is no model.val_step(...) defined
        self.evaluator = Evaluator(steps=val_steps, device=device, progress=progress)

        # General trackers
        self.global_step = 0
        self.val_global_step = 0

        # Callbacks we want to use
        self.checkpointer = none_func

    def checkpoint(self, save_best=False, track="val_loss", method=le):
        self.checkpointer = Checkpointer(
            save_best=save_best, track=track, method=method
        )

    def __call__(
        self, model, train_dataloader, val_dataloader=None, writer=None
    ):
        outputs = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            for i, output in self.epoch_trainer(model, train_dataloader):
                self.global_step += 1,
                if writer is not None: writer.add_scalars("train", output, self.global_step)
            train_outputs = self.epoch_trainer.outputs

            for i, output in self.evaluator(model, val_dataloader)
                self.val_global_step += 1
                if writer is not None: writer.add_scalars("validation", output, self.global_step)
            val_outputs = self.evaluator.outputs  # Even if it doesn't run, it still defines an outputs
            assert val_outputs is not None, "Evaluator should define an outputs upon being called, currently None."

            output = {**train_outputs, **val_outputs}
            self.checkpointer(model, output, epoch=epoch + 1)
            outputs.append(output)
        return outputs



