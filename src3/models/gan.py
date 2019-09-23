import torch.nn as nn
from loss import feature_loss

from abstract_model import Model


class GANModel(Model):
    def __init__(
        self,
        generator,
        critic,
        generator_loss=feature_loss,
        critic_loss=nn.BCEWithLogitsLoss(),
        loss_weights=(1.0, 1.0),
        generator_threshold=float("inf"),
        critic_threshold=float("inf"),
    ):
        super().__init__()
        assert not self.__optimizers
        assert not self.__schedulers

        self.generator = generator
        self.generator_loss = generator_loss
        self.generator_threshold = (
            generator_threshold
        )  # Used to determine when to switch
        self.critic = critic
        self.critic_loss = critic_loss
        self.critic_threshold = critic_threshold
        self.loss_weights = loss_weights
        self.__generator_optimizers = []
        self.__generator_schedulers = []
        self.__critic_optimizers = []
        self.__critic_schedulers = []

        self.last_loss = None

        # Default to generator mode
        self.activate_generator()
        assert self.mode == "generator"

    def activate_generator(self):
        self.mode = "generator"
        self.__optimizers = self.__generator_optimizers
        self.__schedulers = self.__generator_schedulers
        self.threshold = self.generator_threshold
        self.current = self.generator
        self.other = self.critic

    def activate_critic(self):
        self.mode = "critic"
        self.__optimizers = self.__critic_optimizers
        self.__schedulers = self.__critic_schedulers
        self.threshold = self.critic_threshold
        self.current = self.critic
        self.other = self.generator

    def switch(self):
        """Switches training between generator and critic"""
        if self.mode == "generator":
            self.activate_critic()
            assert self.mode == "critic"
        elif self.mode == "critic":
            self.activate_generator()
            assert self.mode == "generator"
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def _loss_g(self, crit_fake_pred, gen_output, target):
        ones = torch.ones_like(crit_fake_pred)
        return self.loss_weights[0] * self.critic_loss(
            crit_fake_pred, ones
        ) + self.loss_weights[1] * self.generator_loss(gen_output, target)

    def _loss_c(self, crit_real_pred, crit_fake_pred):
        ones = torch.ones_like(crit_real_pred)
        zeros = torch.zeros_like(crit_fake_pred)
        return (
            self.critic_loss(crit_real_pred, ones)
            + self.critic_loss(crit_fake_pred, zeros)
        ) / 2

    def train(self):
        # Override the train to only set the model we care about in train mode
        self.current.train()
        self.other.eval()
        return self

    def forward(self, x):
        return self.current(x)

    def train_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]

        # First step
        if self.last_loss is None:
            self.activate_generator()

        # Subsequent steps
        if self.last_loss <= self.threshold:
            # We have to switch here or the trainer will use the wrong optimizers
            self.switch()

        if self.mode == "generator":
            gen_x = self.generator(crap_x)
            crit_fake_pred = self.critic(gen_x)
            gen_loss = self._loss_g(crit_fake_pred, gen_x, good_x)
            self.last_loss = gen_loss  # Use this to figure out when to switch
            return {"loss": gen_loss, "gen_loss": gen_loss}

        if self.mode == "critic":
            crit_real_pred = self.critic(good_x)
            with torch.no_grad():
                gen_x = self.generator(crap_x)
            crit_fake_pred = self.critic(gen_x)
            crit_loss = self._loss_c(crit_real_pred, crit_fake_pred)
            self.last_loss = crit_loss  # Use this to figure out when to switch
            return {"loss": crit_loss, "crit_loss": crit_loss}

        raise ValueError(f"Unknown mode {self.mode}")

    def save_generator(self, path):
        torch.save(self.generator.state_dict, path)

    def save_critic(self, path):
        torch.save(self.critic.state_dict, path)

    def val_step(self, batch):
        crap_x, good_x = batch["input"], batch["target"]

        gen_x = self.generator(crap_x)
        crit_fake_pred = self.critic(gen_x)
        crit_real_pred = self.critic(good_x)

        gen_loss = self._loss_g(crit_fake_pred, gen_x, good_x)
        crit_loss = self._loss_c(crit_real_pred, crit_fake_pred)
        return {
            "loss": gen_loss + crit_loss,
            "gen_loss": gen_loss,
            "crit_loss": crit_loss,
        }

    def predict_step(self, batch):
        crap_x = batch["input"]
        return self.generator(crap_x)
