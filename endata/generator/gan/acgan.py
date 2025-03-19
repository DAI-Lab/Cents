"""
This class is inspired by the synthetic-timeseries-smart-grid GitHub repository:

Repository: https://github.com/vermouth1992/synthetic-time-series-smart-grid
Author: Chi Zhang
License: MIT License

Modifications:
- Hyperparameters and network structure
- Training loop changes
- Changes in conditioning logic

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from endata.datasets.utils import prepare_dataloader
from endata.generator.base_generator import BaseGenerator
from endata.generator.context import ContextModule


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim,
        embedding_dim,
        final_window_length,
        time_series_dims,
        context_module,
        device,
        context_vars=None,
        base_channels=256,
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.time_series_dims = time_series_dims
        self.base_channels = base_channels
        self.device = device

        self.context_vars = context_vars
        self.context_module = context_module

        self.fc = nn.Linear(
            (noise_dim + embedding_dim if self.context_vars else noise_dim),
            self.final_window_length * base_channels,
        ).to(self.device)

        self.conv_transpose_layers = nn.Sequential(
            nn.BatchNorm1d(base_channels).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1
            ).to(self.device),
            nn.BatchNorm1d(base_channels // 2).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 2,
                base_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ).to(self.device),
            nn.BatchNorm1d(base_channels // 4).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 4, time_series_dims, kernel_size=4, stride=2, padding=1
            ).to(self.device),
            nn.Sigmoid().to(self.device),
        ).to(self.device)

    def forward(self, noise, context_vars):
        """
        Forward pass to produce a time series sample.

        Args:
            noise (Tensor): shape (batch_size, noise_dim)
            context_vars (dict): optional dict of context variable Tensors

        Returns:
            generated_time_series (Tensor): shape (batch_size, seq_length, time_series_dims)
            cond_classification_logits (dict): classification logits from the context module
        """
        if context_vars:
            embedding, cond_classification_logits = self.context_module(context_vars)
            x = torch.cat((noise, embedding), dim=1)
        else:
            cond_classification_logits = {}
            x = noise

        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, time_series_dims)

        return x, cond_classification_logits


class Discriminator(nn.Module):
    def __init__(
        self,
        window_length,
        time_series_dims,
        device,
        context_var_n_categories=None,
        base_channels=256,
    ):
        super(Discriminator, self).__init__()
        self.time_series_dims = time_series_dims
        self.window_length = window_length
        self.context_var_n_categories = context_var_n_categories
        self.base_channels = base_channels
        self.device = device

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                time_series_dims, base_channels // 4, kernel_size=4, stride=2, padding=1
            ).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                base_channels // 4,
                base_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ).to(self.device),
            nn.BatchNorm1d(base_channels // 2).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1
            ).to(self.device),
            nn.BatchNorm1d(base_channels).to(self.device),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(self.device)

        self.fc_discriminator = nn.Linear((window_length // 8) * base_channels, 1).to(
            self.device
        )

        self.aux_classifiers = nn.ModuleDict()
        for var_name, num_classes in self.context_var_n_categories.items():
            self.aux_classifiers[var_name] = nn.Linear(
                (window_length // 8) * base_channels, num_classes
            ).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch_size, seq_length, time_series_dims)

        Returns:
            validity (Tensor): shape (batch_size, 1)
            aux_outputs (dict): { var_name: classification_logits }
        """
        x = x.permute(
            0, 2, 1
        )  # Permute to (n_samples, n_dim, seq_length) for conv layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc_discriminator(x))

        aux_outputs = {}
        for var_name, classifier in self.aux_classifiers.items():
            aux_output = classifier(x)  # raw logits
            aux_outputs[var_name] = aux_output

        return validity, aux_outputs


class ACGAN(nn.Module, BaseGenerator):
    def __init__(self, cfg: DictConfig):
        super(ACGAN, self).__init__()
        self.cfg = cfg
        self.code_size = cfg.model.noise_dim
        self.time_series_dims = cfg.dataset.time_series_dims
        self.lr_gen = cfg.model.lr_gen
        self.lr_discr = cfg.model.lr_discr
        self.seq_len = cfg.dataset.seq_len
        self.noise_dim = cfg.model.noise_dim
        self.cond_emb_dim = cfg.model.cond_emb_dim
        self.context_var_n_categories = cfg.dataset.context_vars
        self.device = cfg.device
        self.include_auxiliary_losses = cfg.model.include_auxiliary_losses
        self.cond_loss_weight = cfg.model.cond_loss_weight

        assert (
            self.seq_len % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"

        self.context_module = ContextModule(
            self.context_var_n_categories, self.cond_emb_dim, self.device
        ).to(self.device)

        self.generator = Generator(
            self.noise_dim,
            self.cond_emb_dim,
            self.seq_len,
            self.time_series_dims,
            self.context_module,
            self.device,
            self.context_var_n_categories,
        ).to(self.device)

        self.discriminator = Discriminator(
            self.seq_len,
            self.time_series_dims,
            self.device,
            self.context_var_n_categories,
        ).to(self.device)

        self.adversarial_loss = nn.BCELoss().to(self.device)
        self.auxiliary_loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_discr, betas=(0.5, 0.999)
        )

        if self.cfg.wandb_enabled:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=cfg,
                dir=cfg.run_dir,
            )

    def train_model(self, dataset):
        self.train_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        batch_size = self.cfg.model.batch_size
        num_epoch = self.cfg.model.n_epochs
        train_loader = prepare_dataloader(dataset, batch_size)

        for epoch in range(num_epoch):
            self.current_epoch = epoch + 1

            for _, (time_series_batch, context_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                time_series_batch = time_series_batch.to(self.device)
                context_vars_batch = {
                    name: context_vars_batch[name]
                    for name in self.context_var_n_categories.keys()
                }
                current_batch_size = time_series_batch.size(0)

                self.optimizer_D.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )

                generated_time_series, _ = self.generator(noise, context_vars_batch)
                real_pred, aux_outputs_real = self.discriminator(time_series_batch)
                fake_pred, aux_outputs_fake = self.discriminator(generated_time_series)

                soft_zero, soft_one = 0, 0.95
                d_real_loss = self.adversarial_loss(
                    real_pred, torch.ones_like(real_pred) * soft_one
                )
                d_fake_loss = self.adversarial_loss(
                    fake_pred, torch.ones_like(fake_pred) * soft_zero
                )

                d_loss = d_real_loss + d_fake_loss

                if self.include_auxiliary_losses:
                    for var_name in self.context_var_n_categories.keys():
                        labels = context_vars_batch[var_name].to(self.device)
                        d_loss += self.auxiliary_loss(
                            aux_outputs_real[var_name], labels
                        )
                        d_loss += self.auxiliary_loss(
                            aux_outputs_fake[var_name], labels
                        )

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                gen_categorical_vars = context_vars_batch

                generated_time_series, cond_classification_logits = self.generator(
                    noise, gen_categorical_vars
                )

                validity, aux_outputs = self.discriminator(generated_time_series)
                g_loss = self.adversarial_loss(
                    validity.squeeze(), torch.ones_like(validity.squeeze()) * soft_one
                )

                if self.include_auxiliary_losses:
                    for var_name in self.context_var_n_categories.keys():
                        labels = gen_categorical_vars[var_name].to(self.device)
                        g_loss += self.auxiliary_loss(aux_outputs[var_name], labels)

                cond_loss = 0.0
                for var_name, logits in cond_classification_logits.items():
                    labels = gen_categorical_vars[var_name].to(self.device)
                    cond_loss += self.auxiliary_loss(logits, labels)

                total_generator_loss = g_loss + self.cond_loss_weight * cond_loss

                total_generator_loss.backward()
                self.optimizer_G.step()

                if self.cfg.wandb_enabled:
                    wandb.log(
                        {
                            "Loss/Discriminator": d_loss.item(),
                            "Loss/Generator": g_loss.item(),
                            "Loss/CondClassification": cond_loss.item(),
                            "Loss/Generator_Total": total_generator_loss.item(),
                        }
                    )

            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=self.current_epoch)

    def sample_context_vars(self, dataset, batch_size, random=False):
        """
        Example method for sampling or selecting context var labels.
        Adjust as needed for your data.
        """
        context_vars = {}
        if random:
            for var_name, num_categories in self.context_var_n_categories.items():
                context_vars[var_name] = torch.randint(
                    0,
                    num_categories,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.context_var_n_categories.keys():
                context_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, dtype=torch.long, device=self.device
                )
        return context_vars

    def generate(self, context_vars):
        bs = self.cfg.model.sampling_batch_size
        total = len(next(iter(context_vars.values())))
        generated_samples = []

        for start_idx in range(0, total, bs):
            end_idx = min(start_idx + bs, total)
            batch_context_vars = {
                var_name: var_tensor[start_idx:end_idx]
                for var_name, var_tensor in context_vars.items()
            }
            current_bs = end_idx - start_idx
            noise = torch.randn((current_bs, self.noise_dim)).to(self.device)
            with torch.no_grad():
                generated_data, _ = self.generator(noise, batch_context_vars)
            generated_samples.append(generated_data)

        return torch.cat(generated_samples, dim=0)

    def save(self, path: str = None, epoch: int = None):
        """
        Save the generator and discriminator models, optimizers, and epoch number.

        Args:
            path (str, optional): The file path to save the checkpoint to.
            epoch (int, optional): The current epoch number. Defaults to None.
        """
        if path is None:
            hydra_output_dir = os.path.join(self.cfg.run_dir)

            if not os.path.exists(os.path.join(hydra_output_dir, "checkpoints")):
                os.makedirs(
                    os.path.join(hydra_output_dir, "checkpoints"), exist_ok=True
                )

            path = os.path.join(
                os.path.join(hydra_output_dir, "checkpoints"),
                f"acgan_checkpoint_{epoch if epoch else self.current_epoch}.pt",
            )

        checkpoint = {
            "epoch": epoch if epoch is not None else self.current_epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_state_dict": self.optimizer_D.state_dict(),
            "context_module_state_dict": self.context_module.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Saved ACGAN checkpoint to {path}")

    def load(self, path: str):
        """
        Load the generator and discriminator models, optimizers, and epoch number from a checkpoint file.

        Args:
            path (str): The file path to load the checkpoint from.
        """
        if isinstance(self.device, str):
            if self.device == "cpu":
                map_location = torch.device("cpu")
            else:
                raise ValueError(f"Invalid device string: {self.device}")
        elif isinstance(self.device, int):
            if (
                torch.cuda.is_available()
                and 0 <= self.device < torch.cuda.device_count()
            ):
                map_location = torch.device(f"cuda:{self.device}")
            else:
                raise ValueError(f"Invalid CUDA device index: {self.device}")
        else:
            raise TypeError(
                f"Device should be a string or an integer, but got {type(self.device)}"
            )

        checkpoint = torch.load(path, map_location=map_location)

        if "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            print("Loaded generator state.")
        else:
            raise KeyError("Checkpoint does not contain 'generator_state_dict'.")

        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            print("Loaded discriminator state.")
        else:
            raise KeyError("Checkpoint does not contain 'discriminator_state_dict'.")

        if "optimizer_G_state_dict" in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            print("Loaded generator optimizer state.")
        else:
            print("No generator optimizer state found in checkpoint.")

        if "optimizer_D_state_dict" in checkpoint:
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
            print("Loaded discriminator optimizer state.")
        else:
            print("No discriminator optimizer state found in checkpoint.")

        if "context_module_state_dict" in checkpoint:
            self.context_module.load_state_dict(checkpoint["context_module_state_dict"])
            print("Loaded context module state.")
        else:
            print("No context module state found in checkpoint.")

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
            print(f"Loaded epoch number: {self.current_epoch}")
        else:
            print("No epoch information found in checkpoint.")

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.context_module.to(self.device)
        print(f"ACGAN models moved to {self.device}.")
