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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, noise_dim, day_dim, month_dim, output_dim):
        super(Generator, self).__init__()
        self.day_embedding = nn.Embedding(day_dim, day_dim)
        self.month_embedding = nn.Embedding(month_dim, month_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose1d(
                noise_dim + day_dim + month_dim, 512, kernel_size=4, stride=1, padding=0
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, output_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.Tanh(),
        )

    def forward(self, noise, day, month):
        day_emb = self.day_embedding(day).unsqueeze(-1)
        month_emb = self.month_embedding(month).unsqueeze(-1)
        x = torch.cat((noise, day_emb, month_emb), dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, day_dim, month_dim):
        super(Discriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )

        self.adv_layer = nn.Sequential(nn.Linear(512 * 12, 1), nn.Sigmoid())

        self.aux_layer_day = nn.Sequential(
            nn.Linear(512 * 12, day_dim), nn.Softmax(dim=1)
        )

        self.aux_layer_month = nn.Sequential(
            nn.Linear(512 * 12, month_dim), nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        validity = self.adv_layer(features)
        day_label = self.aux_layer_day(features)
        month_label = self.aux_layer_month(features)
        return validity, day_label, month_label


class ACGAN:
    def __init__(
        self, noise_dim, day_dim, month_dim, output_dim, learning_rate, weight_path
    ):
        self.code_size = noise_dim
        self.batch_size = 64
        self.learning_rate = learning_rate
        self.weight_path = weight_path

        self.generator = Generator(noise_dim, day_dim, month_dim, output_dim).cuda()
        self.discriminator = Discriminator(output_dim, day_dim, month_dim).cuda()

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

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
