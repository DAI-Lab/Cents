import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from data_utils.dataset import prepare_dataloader
from eval.loss import mmd_loss

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, final_window_length, input_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length
        self.input_dim = input_dim

        self.month_embedding = nn.Embedding(12, embedding_dim)
        self.day_embedding = nn.Embedding(7, embedding_dim)

        self.fc = nn.Linear(noise_dim + 2 * embedding_dim, final_window_length * 64)

        self.conv_transpose_layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, input_dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.Sigmoid()
        )

    def forward(self, noise, day_labels, month_labels):
        month_embedded = self.month_embedding(month_labels).view(-1, self.embedding_dim)
        day_embedded = self.day_embedding(day_labels).view(-1, self.embedding_dim)
        
        x = torch.cat((noise, month_embedded, day_embedded), dim=1)
        x = self.fc(x)
        x = x.view(-1, 64, self.final_window_length, 1)
        x = self.conv_transpose_layers(x)
        x = x.squeeze(3)  # Squeeze the last dimension
        
        return x


class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_discriminator = nn.Linear(input_length // 8 * 64, 1)
        self.fc_aux_day = nn.Linear(input_length // 8 * 64, 7)
        self.fc_aux_month = nn.Linear(input_length // 8 * 64, 12)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc_discriminator(x))
        aux_day = self.fc_aux_day(x)
        aux_month = self.fc_aux_month(x)
        return validity, aux_day, aux_month


class ACGAN:
    def __init__(self, noise_dim, embedding_dim, output_dim, learning_rate, weight_path):
        self.code_size = noise_dim
        self.batch_size = 64
        self.learning_rate = learning_rate
        self.weight_path = weight_path

        final_window_length = output_dim 

        self.generator = Generator(noise_dim, embedding_dim, final_window_length, 1).to(device)
        self.discriminator = Discriminator(final_window_length).cuda()

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    def train(self, x_train, x_val, num_epoch=5):
        summary_writer = SummaryWriter()
        self.gen_losses = []
        self.dis_losses = []
        self.mmd_losses = []

        train_loader = prepare_dataloader(x_train, self.batch_size)
        val_loader = prepare_dataloader(x_val, self.batch_size, shuffle=False)

        step = 0

        for epoch in range(num_epoch):
            for i, (
                time_series_batch,
                month_label_batch,
                day_label_batch,
            ) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                time_series_batch, month_label_batch, day_label_batch = (
                    time_series_batch.to(device),
                    month_label_batch.to(device),
                    day_label_batch.to(device),
                )
                noise = torch.randn((self.batch_size, self.code_size)).to(device)
                generated_time_series = self.generator(
                    noise, day_label_batch, month_label_batch
                )

                soft_zero, soft_one = 0, 0.95

                self.optimizer_D.zero_grad()
                real_pred, real_day, real_month = self.discriminator(
                    time_series_batch
                )
                fake_pred, fake_day, fake_month = self.discriminator(
                    generated_time_series.detach()
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
                noise = torch.randn((self.batch_size, self.code_size)).to(device)
                gen_day_labels = torch.randint(0, 7, (self.batch_size,)).to(device)
                gen_month_labels = torch.randint(0, 12, (self.batch_size,)).to(
                    device
                )
                generated_time_series = self.generator(
                    noise, gen_day_labels, gen_month_labels
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

                summary_writer.add_scalars(
                    "data/train_loss",
                    {"gen": g_loss.item(), "dis": d_loss.item()},
                    global_step=step,
                )

                step += 1

            # Validation step
            with torch.no_grad():
                for (
                    time_series_batch,
                    month_label_batch,
                    day_label_batch,
                ) in val_loader:
                    time_series_batch, month_label_batch, day_label_batch = (
                        time_series_batch.to(device),
                        month_label_batch.to(device),
                        day_label_batch.to(device),
                    )
                    x_generated = self.generate(
                        [time_series_batch, month_label_batch, day_label_batch]
                    )
                    mmd_loss_vec = np.zeros(shape=(time_series_batch.shape[-1]))
                    for j in range(time_series_batch.shape[-1]):
                        mmd_loss_vec[j] = mmd_loss(
                            time_series_batch[:, :, j],
                            x_generated[:, :, j],
                            weight=1.0,
                        )
                    summary_writer.add_scalars(
                        "data/mmd_loss",
                        {"load": mmd_loss_vec[0], "pv": mmd_loss_vec[1]},
                        global_step=epoch,
                    )

        self.save_weight()

    def _generate(self, x):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(*x).cpu().numpy()

    def generate(self, labels):
        num_samples = labels[0].shape[0]
        z = np.random.normal(0, 1, size=[num_samples, self.code_size])
        return self._generate(
            [torch.tensor(z, dtype=torch.float32).to(device)]
            + [torch.tensor(l, dtype=torch.long).to(device) for l in labels]
        )

    def generate_by_date(self, num_samples, starting_date_str="2013-01-01"):
        month_labels = np.zeros(shape=(num_samples))
        day_labels = np.zeros(shape=(num_samples))
        starting_date = datetime.datetime.strptime(starting_date_str, date_format_day)
        for i in range(num_samples):
            current_date = starting_date + datetime.timedelta(i)
            month_labels[i] = current_date.month - 1
            day_labels[i] = current_date.weekday()
        return self.generate([month_labels, day_labels])

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
