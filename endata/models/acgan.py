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

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from endata.models.context import ContextModule


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim,
        embedding_dim,
        final_window_length,
        time_series_dims,
        context_module,
        context_vars=None,
        base_channels=256,
    ):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.final_window_length = final_window_length // 8
        self.time_series_dims = time_series_dims
        self.base_channels = base_channels

        self.context_vars = context_vars
        self.context_module = context_module

        self.fc = nn.Linear(
            (noise_dim + embedding_dim if self.context_vars else noise_dim),
            self.final_window_length * base_channels,
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
                base_channels // 4, time_series_dims, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

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
        window_length: int,
        time_series_dims: int,
        context_var_n_categories: dict = None,
        base_channels: int = 256,
    ):
        super().__init__()
        self.time_series_dims = time_series_dims
        self.context_var_n_categories = context_var_n_categories or {}
        self.base_channels = base_channels

        self.conv = nn.Sequential(
            nn.Conv1d(time_series_dims, base_channels // 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 4, base_channels // 2, 4, 2, 1),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 2, base_channels, 4, 2, 1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_real_fake = nn.Linear((window_length // 8) * base_channels, 1)

        self.aux_cls = nn.ModuleDict(
            {
                name: nn.Linear((window_length // 8) * base_channels, n_cls)
                for name, n_cls in self.context_var_n_categories.items()
            }
        )

    def forward(self, x):
        # x: (B, L, C)  ->  (B, C, L)
        x = self.conv(x.permute(0, 2, 1))
        x = x.flatten(1)
        logits = self.fc_real_fake(x)  # (B, 1)

        aux_outputs = {k: head(x) for k, head in self.aux_cls.items()}
        return logits, aux_outputs


class ACGAN(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.automatic_optimization = False

        self.context_module = ContextModule(
            cfg.dataset.context_vars, cfg.model.cond_emb_dim
        )

        self.generator = Generator(
            noise_dim=cfg.model.noise_dim,
            embedding_dim=cfg.model.cond_emb_dim,
            final_window_length=cfg.dataset.seq_len,
            time_series_dims=cfg.dataset.time_series_dims,
            context_module=self.context_module,
            context_vars=cfg.dataset.context_vars,
        )

        self.discriminator = Discriminator(
            window_length=cfg.dataset.seq_len,
            time_series_dims=cfg.dataset.time_series_dims,
            context_var_n_categories=cfg.dataset.context_vars,
        )

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.aux_loss = nn.CrossEntropyLoss()

    def forward(self, noise, context_vars):
        return self.generator(noise, context_vars)

    def configure_optimizers(self):
        opt_G = optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.trainer.optimizer.generator.lr,
            betas=self.cfg.trainer.optimizer.generator.betas,
        )
        opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.trainer.optimizer.discriminator.lr,
            betas=self.cfg.trainer.optimizer.discriminator.betas,
        )
        return [opt_G, opt_D], []

    def training_step(self, batch, batch_idx):
        ts_real, ctx = batch
        bsz = ts_real.size(0)
        noise = torch.randn(bsz, self.cfg.model.noise_dim, device=self.device)

        opt_G, opt_D = self.optimizers()
        smooth_pos, smooth_neg = 0.95, 0.0

        opt_G.zero_grad(set_to_none=True)

        ts_fake, logits_ctx = self.generator(noise, ctx)
        logits_fake, aux_fake = self.discriminator(ts_fake)

        g_adv = self.adv_loss(logits_fake, torch.full_like(logits_fake, smooth_pos))

        g_aux = (
            sum(self.aux_loss(aux_fake[v], ctx[v]) for v in aux_fake)
            if self.cfg.model.include_auxiliary_losses
            else 0.0
        )

        g_ctx = sum(self.aux_loss(logits_ctx[v], ctx[v]) for v in logits_ctx)

        g_total = (
            g_adv + g_aux + self.cfg.model.context_reconstruction_loss_weight * g_ctx
        )
        self.manual_backward(g_total)
        opt_G.step()
        self.log("loss_G", g_total, prog_bar=True, on_step=True)

        opt_D.zero_grad(set_to_none=True)

        # real
        logits_real, aux_real = self.discriminator(ts_real)
        d_real = self.adv_loss(logits_real, torch.full_like(logits_real, smooth_pos))

        # fake (detached)
        logits_fake_det, aux_fake_det = self.discriminator(ts_fake.detach())
        d_fake = self.adv_loss(
            logits_fake_det, torch.full_like(logits_fake_det, smooth_neg)
        )

        d_aux = 0.0
        if self.cfg.model.include_auxiliary_losses:
            for v in aux_real:
                d_aux += self.aux_loss(aux_real[v], ctx[v])
                d_aux += self.aux_loss(aux_fake_det[v], ctx[v])

        d_total = d_real + d_fake + d_aux
        self.manual_backward(d_total)
        opt_D.step()
        self.log("loss_D", d_total, prog_bar=True, on_step=True)

    @torch.no_grad()
    def generate(self, context_vars):
        bs = self.cfg.model.sampling_batch_size
        total = len(next(iter(context_vars.values())))
        out = []
        for s in range(0, total, bs):
            e = min(s + bs, total)
            sub_ctx = {k: v[s:e].to(self.device) for k, v in context_vars.items()}
            noise = torch.randn(e - s, self.cfg.model.noise_dim, device=self.device)
            ts, _ = self.generator(noise, sub_ctx)
            out.append(ts)
        return torch.cat(out, 0)
