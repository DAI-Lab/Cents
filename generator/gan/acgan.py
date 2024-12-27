"""
This class is adapted from the synthetic-timeseries-smart-grid GitHub repository:

Repository: https://github.com/vermouth1992/synthetic-time-series-smart-grid
Author: Chi Zhang
License: MIT License

Modifications:
- Hyperparameters and network structure
- Training loop changes
- Changes in conditioning logic
- Integration of a single post–warm-up GMM fitting for rarity detection

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

from datasets.utils import prepare_dataloader
from generator.conditioning import ConditioningModule


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim,
        embedding_dim,
        final_window_length,
        input_dim,
        conditioning_module,
        device,
        conditioning_var_category_dict=None,
        base_channels=256,
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.device = device

        self.conditioning_var_category_dict = conditioning_var_category_dict
        self.conditioning_module = conditioning_module

        self.fc = nn.Linear(
            (
                noise_dim + embedding_dim
                if self.conditioning_var_category_dict
                else noise_dim
            ),
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
                base_channels // 4, input_dim, kernel_size=4, stride=2, padding=1
            ).to(self.device),
            nn.Sigmoid().to(self.device),
        ).to(self.device)

    def forward(self, noise, conditioning_vars):
        mu, logvar = None, None
        if conditioning_vars:
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=False)
            x = torch.cat((noise, z), dim=1)
        else:
            x = noise
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, n_dim)

        return x, mu, logvar


class Discriminator(nn.Module):
    def __init__(
        self,
        window_length,
        input_dim,
        device,
        conditioning_var_n_categories=None,
        base_channels=256,
    ):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.window_length = window_length
        self.conditioning_var_n_categories = conditioning_var_n_categories
        self.base_channels = base_channels
        self.device = device

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                input_dim, base_channels // 4, kernel_size=4, stride=2, padding=1
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
        for var_name, num_classes in self.conditioning_var_n_categories.items():
            self.aux_classifiers[var_name] = nn.Linear(
                (window_length // 8) * base_channels, num_classes
            ).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  # Permute to (n_samples, n_dim, seq_length) for conv layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc_discriminator(x))

        aux_outputs = {}
        for var_name, classifier in self.aux_classifiers.items():
            aux_output = classifier(x)
            aux_outputs[var_name] = self.softmax(aux_output)

        return validity, aux_outputs


class ACGAN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ACGAN, self).__init__()
        self.cfg = cfg
        self.code_size = cfg.model.noise_dim
        self.input_dim = cfg.dataset.input_dim
        self.lr_gen = cfg.model.lr_gen
        self.lr_discr = cfg.model.lr_discr
        self.seq_len = cfg.dataset.seq_len
        self.noise_dim = cfg.model.noise_dim
        self.cond_emb_dim = cfg.model.cond_emb_dim
        self.conditioning_var_n_categories = cfg.dataset.conditioning_vars
        self.device = cfg.device
        self.warm_up_epochs = cfg.model.warm_up_epochs
        self.sparse_conditioning_loss_weight = cfg.model.sparse_conditioning_loss_weight
        self.kl_weight = cfg.model.kl_weight

        assert (
            self.seq_len % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"

        self.conditioning_module = ConditioningModule(
            self.conditioning_var_n_categories, self.cond_emb_dim, self.device
        ).to(self.device)

        self.generator = Generator(
            self.noise_dim,
            self.cond_emb_dim,
            self.seq_len,
            self.input_dim,
            self.conditioning_module,
            self.device,
            self.conditioning_var_n_categories,
        ).to(self.device)

        self.discriminator = Discriminator(
            self.seq_len,
            self.input_dim,
            self.device,
            self.conditioning_var_n_categories,
        ).to(self.device)

        self.adversarial_loss = nn.BCELoss().to(self.device)
        self.auxiliary_loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_discr, betas=(0.5, 0.999)
        )

        self.gmm_fitted = False

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

            if self.current_epoch > self.warm_up_epochs:
                for param in self.generator.conditioning_module.parameters():
                    param.requires_grad = False

            # ======================================================
            # TRAINING LOOP
            # ======================================================
            for _, (time_series_batch, conditioning_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                time_series_batch = time_series_batch.to(self.device)
                conditioning_vars_batch = {
                    name: conditioning_vars_batch[name]
                    for name in self.conditioning_var_n_categories.keys()
                }

                current_batch_size = time_series_batch.size(0)
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )

                # ---------------------
                # Forward pass G
                # ---------------------
                generated_time_series, mu, logvar = self.generator(
                    noise, conditioning_vars_batch
                )

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                soft_zero, soft_one = 0, 0.95

                real_pred, aux_outputs_real = self.discriminator(time_series_batch)
                fake_pred, aux_outputs_fake = self.discriminator(generated_time_series)

                d_real_loss = self.adversarial_loss(
                    real_pred, torch.ones_like(real_pred) * soft_one
                )
                d_fake_loss = self.adversarial_loss(
                    fake_pred, torch.ones_like(fake_pred) * soft_zero
                )

                if self.cfg.wandb_enabled:
                    wandb.log(
                        {
                            "Loss/Discr_adv": d_real_loss.item() + d_fake_loss.item(),
                        }
                    )

                if self.cfg.model.include_auxiliary_losses:
                    for var_name in self.conditioning_var_n_categories.keys():
                        labels = conditioning_vars_batch[var_name].to(self.device)
                        d_real_loss += self.auxiliary_loss(
                            aux_outputs_real[var_name], labels
                        )
                        d_fake_loss += self.auxiliary_loss(
                            aux_outputs_fake[var_name], labels
                        )

                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                # Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                gen_categorical_vars = self.sample_conditioning_vars(
                    dataset, current_batch_size, random=True
                )
                generated_time_series, mu_g, logvar_g = self.generator(
                    noise, gen_categorical_vars
                )
                validity, aux_outputs = self.discriminator(generated_time_series)

                # Only apply GMM-based rare logic if GMM is fitted
                if self.gmm_fitted:
                    gen_batch_embeddings = mu_g.detach()
                    rare_mask_gen = (
                        self.generator.conditioning_module.is_rare(gen_batch_embeddings)
                        .float()
                        .to(self.device)
                    )
                else:
                    rare_mask_gen = torch.zeros((current_batch_size,)).to(self.device)

                # Adversarial Loss for Generator
                g_loss_rare = self.adversarial_loss(
                    validity.squeeze() * rare_mask_gen,
                    torch.ones_like(validity.squeeze()) * rare_mask_gen * soft_one,
                )
                g_loss_non_rare = self.adversarial_loss(
                    validity.squeeze() * (1 - rare_mask_gen),
                    torch.ones_like(validity.squeeze())
                    * (1 - rare_mask_gen)
                    * soft_one,
                )

                if self.cfg.wandb_enabled:
                    wandb.log(
                        {
                            "Loss/Gen_adv": g_loss_rare.item() + g_loss_non_rare.item(),
                        }
                    )

                if self.cfg.model.include_auxiliary_losses:
                    for var_name in self.conditioning_var_n_categories.keys():
                        labels = gen_categorical_vars[var_name]
                        rare_indices = rare_mask_gen.bool()
                        non_rare_indices = ~rare_mask_gen.bool()

                        if rare_indices.any():
                            g_loss_rare += self.auxiliary_loss(
                                aux_outputs[var_name][rare_indices],
                                labels[rare_indices],
                            )

                        if non_rare_indices.any():
                            g_loss_non_rare += self.auxiliary_loss(
                                aux_outputs[var_name][non_rare_indices],
                                labels[non_rare_indices],
                            )

                _lambda = self.sparse_conditioning_loss_weight
                N_r = rare_mask_gen.sum().item()
                N_nr = (1 - rare_mask_gen).sum().item()
                N = current_batch_size
                g_loss_main = (
                    _lambda * (N_r / N) * g_loss_rare
                    + (1 - _lambda) * (N_nr / N) * g_loss_non_rare
                )

                # KL Divergence (only before warm-up ends)
                if (
                    mu_g is not None
                    and logvar_g is not None
                    and self.current_epoch <= self.warm_up_epochs
                ):
                    kl_loss = self.conditioning_module.kl_divergence(mu_g, logvar_g)

                    if self.cfg.wandb_enabled:
                        wandb.log(
                            {
                                "Loss/KL": kl_loss.item(),
                            }
                        )

                    g_loss = g_loss_main + self.kl_weight * kl_loss
                else:
                    g_loss = g_loss_main

                g_loss.backward()
                self.optimizer_G.step()

                if self.cfg.wandb_enabled:
                    wandb.log(
                        {
                            "Loss/discr_total": d_loss.item(),
                            "Loss/gen_total": g_loss.item(),
                        }
                    )

            # ======================================================
            # After last warmup Epoch - Fit GMM if Warm-Up Just Ended
            # ======================================================
            if self.current_epoch == self.warm_up_epochs and not self.gmm_fitted:
                all_embeddings = []
                full_loader = prepare_dataloader(dataset, batch_size)
                self.generator.conditioning_module.eval()

                with torch.no_grad():
                    for _, (ts_batch, cond_vars_batch) in enumerate(full_loader):
                        cond_vars_batch = {
                            name: cond_vars_batch[name].to(self.device)
                            for name in self.conditioning_var_n_categories.keys()
                        }
                        _, mu_train, _ = self.generator.conditioning_module(
                            cond_vars_batch, sample=False
                        )
                        all_embeddings.append(mu_train.cpu())

                all_embeddings = torch.cat(
                    all_embeddings, dim=0
                )  # shape (N, cond_emb_dim)

                self.generator.conditioning_module.fit_gmm(all_embeddings)
                self.generator.conditioning_module.set_rare_threshold(
                    all_embeddings, fraction=0.1
                )
                self.gmm_fitted = True

            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=self.current_epoch)

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        conditioning_vars = {}
        if random:
            for var_name, num_categories in self.conditioning_var_n_categories.items():
                conditioning_vars[var_name] = torch.randint(
                    0,
                    num_categories,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.conditioning_var_n_categories.keys():
                conditioning_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, dtype=torch.long, device=self.device
                )

        return conditioning_vars

    def generate(self, conditioning_vars):
        num_samples = next(iter(conditioning_vars.values())).shape[0]
        noise = torch.randn((num_samples, self.code_size)).to(self.device)
        with torch.no_grad():
            generated_data, mu, logvar = self.generator(noise, conditioning_vars)
        return generated_data

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
            "conditioning_module_state_dict": self.conditioning_module.state_dict(),
            "gmm_fitted": self.gmm_fitted,
        }
        torch.save(checkpoint, path)
        print(f"Saved ACGAN checkpoint to {path}")

    def load(self, path: str):
        """
        Load the generator and discriminator models, optimizers, and epoch number from a checkpoint file.

        Args:
            path (str): The file path to load the checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)

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

        if "conditioning_module_state_dict" in checkpoint:
            self.conditioning_module.load_state_dict(
                checkpoint["conditioning_module_state_dict"]
            )
            print("Loaded conditioning module state.")
        else:
            print("No conditioning module state found in checkpoint.")

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
            print(f"Loaded epoch number: {self.current_epoch}")
        else:
            print("No epoch information found in checkpoint.")

        if "gmm_fitted" in checkpoint:
            self.gmm_fitted = checkpoint["gmm_fitted"]
        else:
            self.gmm_fitted = False

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.conditioning_module.to(self.device)
        print(f"ACGAN models moved to {self.device}.")
