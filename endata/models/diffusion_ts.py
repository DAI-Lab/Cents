import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from endata.models.context import ContextModule
from endata.models.model_utils import Transformer, default


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.004):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class EMA(nn.Module):
    def __init__(self, model, beta, update_every):
        super().__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model).eval()
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_p.data.mul_(self.beta).add_(model_p.data, alpha=1.0 - self.beta)


class Diffusion_TS(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.seq_len = cfg.dataset.seq_len
        self.time_series_dims = cfg.dataset.time_series_dims
        self.eta = cfg.model.eta
        self.use_ff = cfg.model.use_ff
        self.ff_weight = default(cfg.model.reg_weight, math.sqrt(self.seq_len) / 5)
        self.embedding_dim = cfg.model.cond_emb_dim
        self.context_reconstruction_loss_weight = (
            cfg.model.context_reconstruction_loss_weight
        )
        self.context_module = ContextModule(
            cfg.dataset.context_vars, self.embedding_dim
        )

        self.fc = nn.Linear(
            self.time_series_dims + self.embedding_dim, self.time_series_dims
        )
        self.model = Transformer(
            n_feat=self.time_series_dims + self.embedding_dim,
            n_channel=self.seq_len,
            n_layer_enc=cfg.model.n_layer_enc,
            n_layer_dec=cfg.model.n_layer_dec,
            n_heads=cfg.model.n_heads,
            attn_pdrop=cfg.model.attn_pd,
            resid_pdrop=cfg.model.resid_pd,
            mlp_hidden_times=cfg.model.mlp_hidden_times,
            max_len=self.seq_len,
            n_embd=cfg.model.d_model,
            conv_params=[cfg.model.kernel_size, cfg.model.padding_size],
        )

        if cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.model.n_steps)
        elif cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.model.n_steps)
        else:
            raise ValueError("Unknown beta schedule")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = betas.shape[0]
        self.sampling_timesteps = default(
            cfg.model.sampling_timesteps, self.num_timesteps
        )
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps
        self.loss_type = cfg.model.loss_type

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        pmc1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        pmc2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", pmc1)
        self.register_buffer("posterior_mean_coef2", pmc2)

        lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100
        self.register_buffer("loss_weight", lw)

        if self.loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.recon_loss_fn = F.mse_loss
        else:
            raise ValueError("Invalid loss type")

        self.auxiliary_loss = nn.CrossEntropyLoss()

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t - x0
        ) / self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        pm = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_start
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        pv = self.posterior_variance[t].view(-1, 1, 1)
        plv = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return pm, pv, plv

    def forward(self, x, context_vars):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device)
        embedding, cond_classification_logits = self.context_module(context_vars)
        noise = torch.randn_like(x)
        x_noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        c = torch.cat(
            [x_noisy, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1
        )
        trend, season = self.model(c, t, padding_masks=None)
        x_recon = self.fc(trend + season)
        rec_loss = self.recon_loss_fn(x_recon, x)
        return rec_loss, cond_classification_logits

    def training_step(self, batch, batch_idx):
        ts_batch, cond_batch = batch
        rec_loss, cond_class_logits = self(ts_batch, cond_batch)
        cond_loss = 0.0
        for var_name, logits in cond_class_logits.items():
            labels = cond_batch[var_name]
            cond_loss += self.auxiliary_loss(logits, labels)
        total_loss = rec_loss + self.context_reconstruction_loss_weight * cond_loss
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.cfg.trainer.base_lr, betas=(0.9, 0.96)
        )
        scheduler = ReduceLROnPlateau(optimizer, **self.cfg.trainer.lr_scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def on_train_start(self):
        self.ema = EMA(
            self,
            beta=self.cfg.model.ema_decay,
            update_every=self.cfg.model.ema_update_interval,
            device=self.device,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    @torch.no_grad()
    def model_predictions(self, x, t, embedding):
        c = torch.cat([x, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        x_start = self.fc(trend + season)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @torch.no_grad()
    def p_mean_variance(self, x, t, embedding):
        pred_noise, x_start = self.model_predictions(x, t, embedding)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def p_sample(self, x, t, embedding):
        bt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        pm, pv, plv, _ = self.p_mean_variance(x, bt, embedding)
        noise = torch.randn_like(x) if t > 0 else 0
        return pm + (0.5 * plv).exp() * noise

    @torch.no_grad()
    def sample(self, shape, context_vars):
        x = torch.randn(shape, device=self.device)
        embedding, _ = self.context_module(context_vars)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, embedding)
        return x

    @torch.no_grad()
    def fast_sample(self, shape, context_vars):
        x = torch.randn(shape, device=self.device)
        embedding, _ = self.context_module(context_vars)
        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        for time, time_next in pairs:
            bt = torch.full((x.shape[0],), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x, bt, embedding)
            if time_next < 0:
                x = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                self.eta
                * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    def generate(self, context_vars):
        """
        Public method to generate from the trained model.
        """
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
            shape = (current_bs, self.seq_len, self.time_series_dims)

            with torch.no_grad():
                if getattr(self.cfg.model, "use_ema_sampling", False) and hasattr(
                    self, "ema"
                ):
                    samples = self.ema.ema_model._generate(shape, batch_context_vars)
                else:
                    samples = (
                        self.fast_sample(shape, batch_context_vars)
                        if self.fast_sampling
                        else self.sample(shape, batch_context_vars)
                    )

            generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)
