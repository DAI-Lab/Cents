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

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from datasets.utils import prepare_dataloader, split_dataset
from eval.loss import mmd_loss


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim,
        embedding_dim,
        final_window_length,
        input_dim,
        device,
        base_channels=256,
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.device = device

        self.month_embedding = nn.Embedding(12, embedding_dim).to(self.device)
        self.day_embedding = nn.Embedding(7, embedding_dim).to(self.device)

        self.fc = nn.Linear(
            noise_dim + 2 * embedding_dim, self.final_window_length * base_channels
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

    def forward(self, noise, month_labels, day_labels):
        month_embedded = self.month_embedding(month_labels)
        day_embedded = self.day_embedding(day_labels)

        x = torch.cat((noise, month_embedded, day_embedded), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, seq_length, n_dim)

        return x


class Discriminator(nn.Module):
    def __init__(self, window_length, input_dim, device, base_channels=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.window_length = window_length
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
        self.fc_aux_day = nn.Linear((window_length // 8) * base_channels, 7).to(
            self.device
        )
        self.fc_aux_month = nn.Linear((window_length // 8) * base_channels, 12).to(
            self.device
        )
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        x = x.permute(
            0, 2, 1
        )  # Permute to (n_samples, n_dim, seq_length) for conv layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc_discriminator(x))
        aux_day = self.softmax(self.fc_aux_day(x))
        aux_month = self.softmax(self.fc_aux_month(x))

        return validity, aux_day, aux_month


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
        self.device = opt.device

        assert (
            self.seq_len % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"

        self.generator = Generator(
            self.noise_dim, self.cond_emb_dim, self.seq_len, self.input_dim, self.device
        ).to(self.device)
        self.discriminator = Discriminator(
            self.seq_len, self.input_dim, self.device
        ).to(self.device)

        self.adversarial_loss = nn.BCELoss().to(self.device)
        self.auxiliary_loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_discr, betas=(0.5, 0.999)
        )

    def train_model(self, dataset, validate=False):
        # summary_writer = SummaryWriter()
        self.gen_losses = []
        self.dis_losses = []
        self.mmd_losses = []
        self.gen_adv_losses = []
        self.dis_adv_losses = []

        batch_size = self.opt.batch_size
        num_epoch = self.opt.n_epochs

        if validate:
            x_train, x_val = split_dataset(dataset)
            train_loader = prepare_dataloader(x_train, batch_size)
            val_loader = prepare_dataloader(x_val, batch_size)

        else:
            train_loader = prepare_dataloader(dataset, batch_size)

        step = 0

        for epoch in range(num_epoch):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                current_batch_size = time_series_batch.size(0)
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                generated_time_series = self.generator(
                    noise, month_label_batch, day_label_batch
                )

                soft_zero, soft_one = 0, 0.95

                self.optimizer_D.zero_grad()
                real_pred, real_day, real_month = self.discriminator(time_series_batch)
                fake_pred, fake_day, fake_month = self.discriminator(
                    generated_time_series
                )

                d_real_loss = (
                    self.adversarial_loss(
                        real_pred, torch.ones_like(real_pred) * soft_one
                    )
                    + self.auxiliary_loss(real_day, day_label_batch)
                    + self.auxiliary_loss(real_month, month_label_batch)
                )

                d_fake_loss = (
                    self.adversarial_loss(
                        fake_pred, torch.ones_like(fake_pred) * soft_zero
                    )
                    + self.auxiliary_loss(fake_day, day_label_batch)
                    + self.auxiliary_loss(fake_month, month_label_batch)
                )

                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                gen_day_labels = torch.randint(0, 7, (current_batch_size,)).to(
                    self.device
                )
                gen_month_labels = torch.randint(0, 12, (current_batch_size,)).to(
                    self.device
                )
                generated_time_series = self.generator(
                    noise, gen_month_labels, gen_day_labels
                )

                validity, pred_day, pred_month = self.discriminator(
                    generated_time_series
                )

                g_loss = (
                    self.adversarial_loss(
                        validity, torch.ones_like(validity) * soft_one
                    )
                    + self.auxiliary_loss(pred_day, gen_day_labels)
                    + self.auxiliary_loss(pred_month, gen_month_labels)
                )

                g_loss.backward()
                self.optimizer_G.step()

            # Validation step
            if validate:
                with torch.no_grad():
                    total_mmd_loss = np.zeros(shape=(self.input_dim,))
                    num_batches = 0
                    for (
                        time_series_batch,
                        month_label_batch,
                        day_label_batch,
                    ) in val_loader:
                        x_generated = self.generate(
                            month_labels=month_label_batch, day_labels=day_label_batch
                        )
                        mmd_values = np.zeros(
                            shape=(time_series_batch.shape[0], self.input_dim)
                        )

                        for dim in range(self.input_dim):
                            mmd_values[:, dim] = mmd_loss(
                                time_series_batch[:, :, dim].cpu().numpy(),
                                x_generated[:, :, dim].cpu().numpy(),
                            )

                        batch_mmd_loss = np.mean(mmd_values, axis=0)
                        total_mmd_loss += batch_mmd_loss
                        num_batches += 1

                    mean_mmd_loss = total_mmd_loss / num_batches
                    print(
                        f"Epoch [{epoch + 1}/{num_epoch}], Mean MMD Loss: {mean_mmd_loss}"
                    )

    def _generate(self, x):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(*x)

    def generate(self, day_labels, month_labels):
        assert (
            day_labels.shape == month_labels.shape
        ), "Number of weekday and month labels must be equal!"

        num_samples = day_labels.shape[0]
        noise = torch.randn((num_samples, self.code_size)).to(self.device)
        return self._generate(
            [
                noise,
                month_labels.clone().to(self.device),
                day_labels.clone().to(self.device),
            ]
        )
