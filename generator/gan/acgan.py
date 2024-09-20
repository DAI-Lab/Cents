"""
This class is adapted from the synthetic-timeseries-smart-grid GitHub repository:

Repository: https://github.com/vermouth1992/synthetic-time-series-smart-grid
Author: Chi Zhang
License: MIT License

Modifications:
- Hyperparameters and network structure
- Training loop changes
- Changes in conditioning logic

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

        self.conditioning_module = ConditioningModule(
            self.categorical_dims, self.embedding_dim, self.device
        ).to(self.device)

        self.fc = nn.Linear(
            # noise_dim + 2 * self.embedding_dim, self.final_window_length * base_channels
            noise_dim + self.embedding_dim,
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
        conditioning_vector = self.conditioning_module(conditioning_vars)
        x = torch.cat((noise, conditioning_vector), dim=1)
        x = self.fc(x)
        x = x.view(-1, self.base_channels, self.final_window_length)
        x = self.conv_transpose_layers(x)
        x = x.permute(0, 2, 1)  # Permute to (batch_size, seq_length, n_dim)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        window_length,
        input_dim,
        device,
        categorical_dims=None,
        base_channels=256,
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
        ).to(self.device)

        self.fc_discriminator = nn.Linear((window_length // 8) * base_channels, 1).to(
            self.device
        )

        # Auxiliary classifiers for each conditioning variable
        self.aux_classifiers = nn.ModuleDict()
        for var_name, num_classes in self.categorical_dims.items():
            self.aux_classifiers[var_name] = nn.Linear(
                (window_length // 8) * base_channels, num_classes
            ).to(self.device)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        validity = self.sigmoid(self.fc_discriminator(x))

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
        self.device = opt.device
        self.embedding_dim = opt.cond_emb_dim
        self.categorical_dims = opt.categorical_dims

        assert (
            self.seq_len % 8 == 0
        ), "window_length must be a multiple of 8 in this architecture!"

        self.generator = Generator(
            self.noise_dim,
            self.embedding_dim,
            self.seq_len,
            self.input_dim,
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

        for epoch in range(num_epoch):
            for i, (time_series_batch, conditioning_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                conditioning_vars_batch = {
                    name: conditioning_vars_batch[name]
                    for name in self.categorical_dims.keys()
                }
                current_batch_size = time_series_batch.size(0)
                time_series_batch = time_series_batch.to(self.device)

                # Generate noise
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )

                # Generate fake data
                generated_time_series = self.generator(noise, conditioning_vars_batch)

                soft_zero, soft_one = 0, 0.95

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Real data
                validity_real, aux_outputs_real = self.discriminator(time_series_batch)
                d_real_adv_loss = self.adversarial_loss(
                    validity_real, torch.ones_like(validity_real) * soft_one
                )
                d_real_aux_loss = 0
                for var_name in self.categorical_dims.keys():
                    labels = conditioning_vars_batch[var_name].to(self.device)
                    d_real_aux_loss += self.auxiliary_loss(
                        aux_outputs_real[var_name], labels
                    )

                # Fake data
                validity_fake, aux_outputs_fake = self.discriminator(
                    generated_time_series.detach()
                )
                d_fake_adv_loss = self.adversarial_loss(
                    validity_fake, torch.zeros_like(validity_fake) * soft_zero
                )
                d_fake_aux_loss = 0
                for var_name in self.categorical_dims.keys():
                    labels = conditioning_vars_batch[var_name].to(self.device)
                    d_fake_aux_loss += self.auxiliary_loss(
                        aux_outputs_fake[var_name], labels
                    )

                d_loss = 0.5 * (d_real_adv_loss + d_fake_adv_loss) + 0.5 * (
                    d_real_aux_loss + d_fake_aux_loss
                )
                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()

                # Generate new conditioning variables
                gen_categorical_vars = self.sample_conditioning_vars(
                    dataset, current_batch_size, random=True
                )

                # Generate fake data
                noise = torch.randn((current_batch_size, self.code_size)).to(
                    self.device
                )
                generated_time_series = self.generator(noise, gen_categorical_vars)

                # Discriminator's response
                validity, aux_outputs = self.discriminator(generated_time_series)

                # Generator adversarial loss
                g_adv_loss = self.adversarial_loss(
                    validity, torch.ones_like(validity) * soft_one
                )

                # Auxiliary losses for generator
                g_aux_loss = 0
                for var_name in self.categorical_dims.keys():
                    labels = gen_categorical_vars[var_name]
                    g_aux_loss += self.auxiliary_loss(aux_outputs[var_name], labels)

                # Total generator loss
                g_loss = g_adv_loss + g_aux_loss
                g_loss.backward()
                self.optimizer_G.step()

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        conditioning_vars = {}
        if random:
            for var_name, num_categories in self.categorical_dims.items():
                conditioning_vars[var_name] = torch.randint(
                    0, num_categories, (batch_size,), device=self.device
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.categorical_dims.keys():
                conditioning_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, device=self.device
                )

        return conditioning_vars

    def generate(self, conditioning_vars):
        num_samples = next(iter(conditioning_vars.values())).shape[0]
        noise = torch.randn((num_samples, self.code_size)).to(self.device)
        with torch.no_grad():
            generated_data = self.generator(noise, conditioning_vars)
        return generated_data
