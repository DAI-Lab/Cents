import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from data_utils.dataset import prepare_dataloader
from eval.loss import mmd_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(
        self, noise_dim, embedding_dim, final_window_length, input_dim, base_channels=64
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.input_dim = input_dim
        self.base_channels = base_channels

        self.month_embedding = nn.Embedding(12, embedding_dim)
        self.day_embedding = nn.Embedding(7, embedding_dim)

        self.fc = nn.Linear(
            noise_dim + 2 * embedding_dim, self.final_window_length * base_channels
        )

        self.conv_transpose_layers = nn.Sequential(
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 2,
                base_channels // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels // 4, input_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, noise, month_labels, day_labels):
        month_embedded = (
            self.month_embedding(month_labels).view(-1, self.embedding_dim).to(device)
        )
        day_embedded = (
            self.day_embedding(day_labels).view(-1, self.embedding_dim).to(device)
        )

        x = torch.cat((noise, month_embedded, day_embedded), dim=1).to(device)
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, seq_length, n_dim)

        return x


class Discriminator(nn.Module):
    def __init__(self, window_length, input_dim, base_channels=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.window_length = window_length
        self.base_channels = base_channels

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                input_dim, base_channels // 4, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                base_channels // 4,
                base_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_discriminator = nn.Linear((window_length // 8) * base_channels, 1)
        self.fc_aux_day = nn.Linear((window_length // 8) * base_channels, 7)
        self.fc_aux_month = nn.Linear((window_length // 8) * base_channels, 12)
        self.softmax = nn.Softmax(dim=1)

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
    def __init__(
        self,
        input_dim,
        noise_dim,
        embedding_dim,
        window_length,
        learning_rate,
        weight_path,
    ):
        self.code_size = noise_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_path = weight_path

        assert (
            window_length % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"
        final_window_length = window_length
        self.window_length = final_window_length

        self.generator = Generator(
            noise_dim, embedding_dim, final_window_length, input_dim
        ).to(device)
        self.discriminator = Discriminator(window_length, input_dim).to(device)

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=1e-6,
        )

    def train(self, x_train, x_val, batch_size=32, num_epoch=5):
        summary_writer = SummaryWriter()
        self.gen_losses = []
        self.dis_losses = []
        self.mmd_losses = []
        self.gen_adv_losses = []
        self.dis_adv_losses = []

        train_loader = prepare_dataloader(x_train, batch_size)
        val_loader = prepare_dataloader(x_val, batch_size)

        step = 0

        for epoch in range(num_epoch):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                current_batch_size = time_series_batch.size(0)
                noise = torch.randn((current_batch_size, self.code_size)).to(device)
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

                d_fake_loss_adv = self.adversarial_loss(
                    fake_pred, torch.ones_like(fake_pred) * soft_zero
                )

                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                noise = torch.randn((current_batch_size, self.code_size)).to(device)
                gen_day_labels = torch.randint(0, 7, (current_batch_size,)).to(device)
                gen_month_labels = torch.randint(0, 12, (current_batch_size,)).to(
                    device
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

                g_loss_adv = self.adversarial_loss(
                    validity, torch.ones_like(validity) * soft_one
                )

                g_loss.backward()

                self.optimizer_G.step()

                summary_writer.add_scalars(
                    "data/train_loss",
                    {"gen": g_loss.item(), "dis": d_loss.item()},
                    global_step=step,
                )

                summary_writer.add_scalars(
                    "data/adv_loss",
                    {"dis": d_fake_loss_adv.item(), "gen": g_loss_adv.item()},
                    global_step=step,
                )

                step += 1

            # Validation step
            with torch.no_grad():
                total_mmd_loss = np.zeros(shape=(self.input_dim,))
                num_batches = 0
                for time_series_batch, month_label_batch, day_label_batch in val_loader:
                    time_series_batch, month_label_batch, day_label_batch = (
                        time_series_batch,
                        month_label_batch,
                        day_label_batch,
                    )
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
                summary_writer.add_scalars(
                    "data/mean_mmd_loss",
                    {
                        "grid": total_mmd_loss[0],
                        "solar": (
                            total_mmd_loss[1] if total_mmd_loss.shape[0] > 1 else 0
                        ),
                    },
                    global_step=epoch,
                )
                print(
                    f"Epoch [{epoch + 1}/{num_epoch}], Mean MMD Loss: {mean_mmd_loss}"
                )

        # self.save_weight()

    def _generate(self, x):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(*x)

    def generate(self, day_labels, month_labels):
        assert (
            day_labels.shape == month_labels.shape
        ), "Number of weekday and month labels must be equal!"

        num_samples = day_labels.shape[0]
        noise = torch.randn((num_samples, self.code_size)).to(device)
        return self._generate([noise] + [month_labels.clone()] + [day_labels.clone()])

    def save_weight(self):
        torch.save(
            self.generator.state_dict(), self.weight_path + "_acgan_generator.pth"
        )
        torch.save(
            self.discriminator.state_dict(),
            self.weight_path + "_acgan_discriminator.pth",
        )

    def load_weight(self):
        self.generator.load_state_dict(
            torch.load(self.weight_path + "_acgan_generator.pth")
        )
        self.discriminator.load_state_dict(
            torch.load(self.weight_path + "_acgan_discriminator.pth")
        )
