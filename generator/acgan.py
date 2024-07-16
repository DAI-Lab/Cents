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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, final_window_length, input_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.input_dim = input_dim

        self.month_embedding = nn.Embedding(12, embedding_dim)
        self.day_embedding = nn.Embedding(7, embedding_dim)

        self.fc = nn.Linear(
            noise_dim + 2 * embedding_dim, self.final_window_length * 64
        )

        self.conv_transpose_layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                16, input_dim, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)
            ),
            nn.Sigmoid(),
        )

    def forward(self, noise, day_labels, month_labels):
        month_embedded = (
            self.month_embedding(month_labels).view(-1, self.embedding_dim).to(device)
        )
        day_embedded = (
            self.day_embedding(day_labels).view(-1, self.embedding_dim).to(device)
        )

        x = torch.cat((noise, month_embedded, day_embedded), dim=1)
        x = self.fc(x)
        x = x.view(-1, 64, self.final_window_length, 1)
        x = self.conv_transpose_layers(x)
        x = x.squeeze(3)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_length, input_dim=1):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=4, stride=2, padding=1),
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = torch.sigmoid(self.fc_discriminator(x))
        aux_day = self.softmax(self.fc_aux_day(x))
        aux_month = self.softmax(self.fc_aux_month(x))

        if (
            torch.isnan(validity).any()
            or torch.isnan(aux_day).any()
            or torch.isnan(aux_month).any()
        ):
            print("NaN detected in Discriminator output")
            print(f"x stats: min={x.min()}, max={x.max()}, mean={x.mean()}")

        return validity, aux_day, aux_month


class ACGAN:
    def __init__(
        self,
        input_dim,
        noise_dim,
        embedding_dim,
        output_dim,
        learning_rate,
        weight_path,
    ):
        self.code_size = noise_dim
        self.learning_rate = learning_rate
        self.weight_path = weight_path

        assert (
            output_dim % 8 == 0
        ), "output_dim must be a multiple of 8 in this architecture!"
        final_window_length = output_dim

        self.generator = Generator(
            noise_dim, embedding_dim, final_window_length, input_dim
        ).to(device)
        self.discriminator = Discriminator(output_dim).to(device)

        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

    def train(self, x_train, x_val, batch_size=32, num_epoch=5):
        summary_writer = SummaryWriter()
        self.gen_losses = []
        self.dis_losses = []
        self.mmd_losses = []

        train_loader = prepare_dataloader(x_train, batch_size)
        val_loader = prepare_dataloader(x_val, batch_size)

        step = 0

        for epoch in range(num_epoch):
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                time_series_batch, month_label_batch, day_label_batch = (
                    time_series_batch.to(device),
                    month_label_batch.to(device),
                    day_label_batch.to(device),
                )
                current_batch_size = time_series_batch.size(0)
                noise = torch.randn((current_batch_size, self.code_size)).to(device)
                generated_time_series = self.generator(
                    noise, day_label_batch, month_label_batch
                )

                soft_zero, soft_one = 0, 0.95

                self.optimizer_D.zero_grad()
                real_pred, real_day, real_month = self.discriminator(time_series_batch)
                fake_pred, fake_day, fake_month = self.discriminator(
                    generated_time_series.detach().squeeze()
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
                # print("Discriminator gradients:")
                # for name, param in self.discriminator.named_parameters():
                #   if param.grad is not None:
                #      print(f"{name} - grad mean: {param.grad.mean()}, grad std: {param.grad.std()}")

                self.optimizer_D.step()
                for _ in range(10):
                    self.optimizer_G.zero_grad()
                    noise = torch.randn((current_batch_size, self.code_size)).to(device)
                    gen_day_labels = torch.randint(0, 7, (current_batch_size,)).to(
                        device
                    )
                    gen_month_labels = torch.randint(0, 12, (current_batch_size,)).to(
                        device
                    )
                    generated_time_series = self.generator(
                        noise, gen_day_labels, gen_month_labels
                    )

                    validity, pred_day, pred_month = self.discriminator(
                        generated_time_series.detach().squeeze()
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
                for time_series_batch, month_label_batch, day_label_batch in val_loader:
                    time_series_batch, month_label_batch, day_label_batch = (
                        time_series_batch.to(device),
                        month_label_batch.to(device),
                        day_label_batch.to(device),
                    )
                    x_generated = self.generate([day_label_batch, month_label_batch])
                    mmd = mmd_loss(
                        time_series_batch.cpu().numpy(), x_generated.squeeze()
                    )
                    summary_writer.add_scalars(
                        "data/mmd_loss",
                        {"load": mmd},
                        global_step=epoch,
                    )

                print(f"Epoch [{epoch + 1}/{num_epoch}], MMD Loss: {mmd}")

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
            + [l.clone().detach() for l in labels]
        )

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
