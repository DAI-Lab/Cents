"""
This class is adapted/taken from the synthetic-timeseries-smart-grid GitHub repository:

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
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
        categorical_dims=None,
        base_channels=256,
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.device = device

        self.categorical_dims = categorical_dims

        self.conditioning_module = conditioning_module

        self.fc = nn.Linear(
            noise_dim + embedding_dim, self.final_window_length * base_channels
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
        conditioning_vector = self.conditioning_module(conditioning_vars)
        x = torch.cat((noise, conditioning_vector), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, seq_length, n_dim)

        return x


class Discriminator(nn.Module):
    def __init__(
        self, window_length, input_dim, device, categorical_dims=None, base_channels=256
    ):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.window_length = window_length
        self.categorical_dims = categorical_dims
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
        for var_name, num_classes in self.categorical_dims.items():
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


class ACGAN:
    def __init__(self, opt):
        self.opt = opt
        self.code_size = opt.noise_dim
        self.input_dim = opt.input_dim
        self.lr_gen = opt.lr_gen
        self.lr_discr = opt.lr_discr
        self.seq_len = opt.seq_len
        self.noise_dim = opt.noise_dim
        self.cond_emb_dim = opt.cond_emb_dim
        self.categorical_dims = opt.categorical_dims
        self.device = opt.device
        self.warm_up_epochs = opt.warm_up_epochs
        self.sparse_conditioning_loss_weight = opt.sparse_conditioning_loss_weight

        self.writer = SummaryWriter(log_dir=os.path.join("runs", "acgan"))

        assert (
            self.seq_len % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"

        self.conditioning_module = ConditioningModule(
            self.categorical_dims, self.cond_emb_dim, self.device
        ).to(self.device)

        self.generator = Generator(
            self.noise_dim,
            self.cond_emb_dim,
            self.seq_len,
            self.input_dim,
            self.conditioning_module,
            self.device,
            self.categorical_dims,
        ).to(self.device)
        self.discriminator = Discriminator(
            self.seq_len, self.input_dim, self.device, self.categorical_dims
        ).to(self.device)

        self.adversarial_loss = nn.BCELoss().to(self.device)
        self.auxiliary_loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_discr, betas=(0.5, 0.999)
        )

    def train_model(self, dataset):
        batch_size = self.opt.batch_size
        num_epoch = self.opt.n_epochs
        train_loader = prepare_dataloader(dataset, batch_size)

        previous_mean_embedding = None
        previous_embedding_covariance = None

        for epoch in range(num_epoch):
            for batch_index, (time_series_batch, conditioning_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                time_series_batch = time_series_batch.to(self.device)
                conditioning_vars_batch = {
                    name: conditioning_vars_batch[name]
                    for name in self.categorical_dims.keys()
                }

                current_batch_size = time_series_batch.size(0)
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                generated_time_series = self.generator(
                    noise, conditioning_vars_batch
                ).to(self.device)

                soft_zero, soft_one = 0, 0.95

                rare_mask = torch.zeros((current_batch_size,)).to(self.device)

                if epoch > self.warm_up_epochs:

                    batch_embeddings = self.generator.conditioning_module(
                        conditioning_vars_batch
                    )
                    self.generator.conditioning_module.update_running_statistics(
                        batch_embeddings
                    )

                    if self.opt.freeze_cond_after_warmup:
                        for param in self.generator.conditioning_module.parameters():
                            param.requires_grad = False  # if specified, freeze conditioning module training

                    rare_mask = (
                        self.generator.conditioning_module.is_rare(batch_embeddings)
                        .to(self.device)
                        .float()
                    )

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                real_pred, aux_outputs_real = self.discriminator(time_series_batch)
                fake_pred, aux_outputs_fake = self.discriminator(generated_time_series)

                d_real_loss = self.adversarial_loss(
                    real_pred, torch.ones_like(real_pred) * soft_one
                )
                d_fake_loss = self.adversarial_loss(
                    fake_pred, torch.ones_like(fake_pred) * soft_zero
                )
                if self.opt.include_auxiliary_losses:
                    for var_name in self.categorical_dims.keys():
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
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                gen_categorical_vars = self.sample_conditioning_vars(
                    dataset, current_batch_size, random=True
                )
                generated_time_series = self.generator(noise, gen_categorical_vars)

                validity, aux_outputs = self.discriminator(generated_time_series)

                g_loss_rare = self.adversarial_loss(
                    validity.squeeze() * rare_mask,
                    torch.ones_like(validity.squeeze()) * rare_mask * soft_one,
                )

                g_loss_non_rare = self.adversarial_loss(
                    validity.squeeze() * (torch.logical_not(rare_mask)),
                    torch.ones_like(validity.squeeze())
                    * torch.logical_not(rare_mask)
                    * soft_one,
                )
                _lambda = self.sparse_conditioning_loss_weight
                N_r = rare_mask.sum().item()
                N_nr = (torch.logical_not(rare_mask)).sum().item()
                N = current_batch_size
                g_loss = (
                    _lambda * (N_r / N) * g_loss_rare
                    + (1 - _lambda) * (N_nr / N) * g_loss_non_rare
                )

                if self.opt.include_auxiliary_losses:
                    for var_name in self.categorical_dims.keys():
                        labels = gen_categorical_vars[var_name]
                        g_loss += self.auxiliary_loss(aux_outputs[var_name], labels)

                g_loss.backward()
                self.optimizer_G.step()

                # -------------------
                # TensorBoard Logging
                # -------------------
                global_step = epoch * len(train_loader) + batch_index

                # Log overall losses for both generator and discriminator
                # self.writer.add_scalars('Losses', {'Discriminator': d_loss.item(), 'Generator': g_loss.item()}, global_step)

            # End of epoch logging
            if epoch > self.warm_up_epochs:

                self.generator.conditioning_module.log_embedding_statistics(
                    epoch,
                    self.writer,
                    previous_mean_embedding,
                    previous_embedding_covariance,
                    batch_embeddings,
                )
                previous_mean_embedding = (
                    self.generator.conditioning_module.mean_embedding.clone()
                )
                previous_embedding_covariance = (
                    self.generator.conditioning_module.cov_embedding.clone()
                )

        self.writer.close()

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        conditioning_vars = {}
        if random:
            for var_name, num_categories in self.categorical_dims.items():
                conditioning_vars[var_name] = torch.randint(
                    0,
                    num_categories,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.categorical_dims.keys():
                conditioning_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, dtype=torch.long, device=self.device
                )

        return conditioning_vars

    def generate(self, conditioning_vars):
        num_samples = next(iter(conditioning_vars.values())).shape[0]
        noise = torch.randn((num_samples, self.code_size)).to(self.device)
        with torch.no_grad():
            generated_data = self.generator(noise, conditioning_vars)
        return generated_data
