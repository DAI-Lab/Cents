"""
This class is adapted/taken from the Diffusion_TS GitHub repository:

Repository: https://github.com/Y-debug-sys/Diffusion-TS
Author: Xinyu Yuan
License: MIT License

Modifications:
- Conditioning and sampling logic
- Added further functions and removed unused functionality

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import copy
import math
import os
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from generator.conditioning import ConditioningModule
from generator.diffusion_ts.model_utils import default, extract, identity
from generator.diffusion_ts.transformer import Transformer


def linear_beta_schedule(timesteps, device):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(
        device
    )


def cosine_beta_schedule(timesteps, device, s=0.004):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32).to(device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to(device)


class Diffusion_TS(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Diffusion_TS, self).__init__()
        self.cfg = cfg
        self.eta, self.use_ff = cfg.model.eta, cfg.model.use_ff
        self.seq_len = cfg.dataset.seq_len
        self.input_dim = cfg.dataset.input_dim
        self.ff_weight = default(
            cfg.model.reg_weight, math.sqrt(cfg.dataset.seq_len) / 5
        )
        self.device = cfg.device
        self.embedding_dim = cfg.model.cond_emb_dim
        self.categorical_dims = cfg.dataset.conditioning_vars
        self.warm_up_epochs = cfg.model.warm_up_epochs
        self.sparse_conditioning_loss_weight = cfg.model.sparse_conditioning_loss_weight
        self.kl_weight = cfg.model.kl_weight  # Ensure you have this in your config

        self.conditioning_module = ConditioningModule(
            self.categorical_dims, self.embedding_dim, self.device
        ).to(self.device)

        self.fc = nn.Linear(self.input_dim + self.embedding_dim, self.input_dim)

        self.model = Transformer(
            n_feat=self.input_dim + self.embedding_dim,
            n_channel=self.cfg.dataset.seq_len,
            n_layer_enc=self.cfg.model.n_layer_enc,
            n_layer_dec=self.cfg.model.n_layer_dec,
            n_heads=self.cfg.model.n_heads,
            attn_pdrop=self.cfg.model.attn_pd,
            resid_pdrop=self.cfg.model.resid_pd,
            mlp_hidden_times=self.cfg.model.mlp_hidden_times,
            max_len=self.cfg.dataset.seq_len,
            n_embd=self.cfg.model.d_model,
            conv_params=[self.cfg.model.kernel_size, self.cfg.model.padding_size],
        )

        # Beta schedule
        if self.cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(self.cfg.model.n_steps, self.device)
        elif self.cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.cfg.model.n_steps, self.device)
        else:
            raise ValueError(f"unknown beta schedule {self.cfg.model.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(
            self.device
        )

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = self.cfg.model.loss_type

        # Sampling timesteps
        self.sampling_timesteps = default(self.cfg.model.sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(self.device)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None, conditioning_vars=None):
        if conditioning_vars is not None:
            # Deterministic embeddings for stability
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=False)
            conditioning_vector = mu.unsqueeze(1).repeat(1, self.seq_len, 1)
            x = torch.cat([x, conditioning_vector], dim=-1)
        else:
            mu, logvar = None, None

        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        out = self.fc(model_output)
        return out, mu, logvar

    def model_predictions(self, x, t, conditioning_vars, clip_x_start=False):
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start, mu, logvar = self.output(x, t, conditioning_vars=conditioning_vars)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_sample(self, x, t: int, conditioning_vars, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            conditioning_vars=conditioning_vars,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_mean_variance(self, x, t, conditioning_vars, clip_denoised=True):
        _, x_start = self.model_predictions(
            x, t, conditioning_vars, clip_x_start=clip_denoised
        )
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def sample(self, shape, conditioning_vars):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t, conditioning_vars)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, conditioning_vars, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                img, time_cond, conditioning_vars, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

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

    def _generate(self, shape, conditioning_vars):
        self.eval()
        with torch.no_grad():
            samples, mu, logvar = (
                self.fast_sample(shape, conditioning_vars)
                if self.fast_sampling
                else self.sample(shape, conditioning_vars)
            )
            return samples

    def generate(self, conditioning_vars):
        num_samples = len(conditioning_vars[list(conditioning_vars.keys())[0]])
        shape = (num_samples, self.seq_len, self.input_dim)

        if self.cfg.model.use_ema_sampling:
            return self.ema.ema_model._generate(shape, conditioning_vars)
        else:
            return self._generate(shape, conditioning_vars)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(
        self,
        x_start,
        t,
        target=None,
        noise=None,
        padding_masks=None,
        conditioning_vars=None,
    ):
        # Obtain mu, logvar from conditioning module
        with torch.set_grad_enabled(self.training):
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=False)

        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_cat = torch.cat([x, mu.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(x_cat, t, padding_masks=padding_masks)
        model_out = self.fc(trend + season)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean(), mu, logvar

    def forward(self, x, conditioning_vars=None, **kwargs):
        b, seq_len, input_dim = x.shape
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        loss, mu, logvar = self._train_loss(
            x, t, conditioning_vars=conditioning_vars, **kwargs
        )
        return loss, mu, logvar

    def train_model(self, train_dataset):
        self.train_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.train()
        self.to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=self.cfg.dataset.shuffle,
            drop_last=True,
        )

        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.model.base_lr,
            betas=[0.9, 0.96],
        )
        self.ema = EMA(
            self,
            beta=self.cfg.model.ema_decay,
            update_every=self.cfg.model.ema_update_interval,
            device=self.device,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, **self.cfg.model.lr_scheduler_params
        )

        for epoch in range(self.cfg.model.n_epochs):
            self.current_epoch = epoch + 1
            total_loss = 0.0

            # Freeze conditioning module after warm-up
            if (
                self.current_epoch > self.warm_up_epochs
                and self.cfg.model.freeze_cond_after_warmup
            ):
                for param in self.conditioning_module.parameters():
                    param.requires_grad = False

            for i, (time_series_batch, conditioning_vars_batch) in enumerate(
                train_loader
            ):
                time_series_batch = time_series_batch.to(self.device)
                for key in conditioning_vars_batch:
                    conditioning_vars_batch[key] = conditioning_vars_batch[key].to(
                        self.device
                    )

                current_batch_size = time_series_batch.size(0)

                if self.current_epoch <= self.warm_up_epochs:
                    # Warm-up period: no rarity, update running stats, apply KL
                    loss, mu, logvar = self(
                        time_series_batch, conditioning_vars=conditioning_vars_batch
                    )
                    # Update running statistics
                    self.conditioning_module.update_running_statistics(mu.detach())

                    # KL loss during warm-up
                    if mu is not None and logvar is not None:
                        kl_loss = self.conditioning_module.kl_divergence(mu, logvar)
                        loss = loss + self.kl_weight * kl_loss

                else:
                    # Post warm-up: freeze conditioning module, compute rarity
                    # No KL loss here
                    with torch.no_grad():
                        _, mu, _ = self(
                            time_series_batch, conditioning_vars=conditioning_vars_batch
                        )
                        # Do not backprop through this call
                        mu_detach = mu.detach()
                        rare_mask = (
                            self.conditioning_module.is_rare(mu_detach)
                            .float()
                            .to(self.device)
                        )

                    # Separate rare and non-rare
                    rare_indices = (rare_mask == 1.0).nonzero(as_tuple=True)[0]
                    non_rare_indices = (rare_mask == 0.0).nonzero(as_tuple=True)[0]

                    loss_rare = torch.tensor(0.0, device=self.device)
                    loss_non_rare = torch.tensor(0.0, device=self.device)

                    if len(rare_indices) > 0:
                        ts_rare = time_series_batch[rare_indices]
                        cond_rare = {
                            k: v[rare_indices]
                            for k, v in conditioning_vars_batch.items()
                        }
                        loss_rare, _, _ = self(ts_rare, conditioning_vars=cond_rare)

                    if len(non_rare_indices) > 0:
                        ts_non_rare = time_series_batch[non_rare_indices]
                        cond_non_rare = {
                            k: v[non_rare_indices]
                            for k, v in conditioning_vars_batch.items()
                        }
                        loss_non_rare, _, _ = self(
                            ts_non_rare, conditioning_vars=cond_non_rare
                        )

                    N_r = rare_mask.sum().item()
                    N_nr = (1 - rare_mask).sum().item()
                    N = current_batch_size
                    _lambda = self.sparse_conditioning_loss_weight
                    loss = (
                        _lambda * (N_r / N) * loss_rare
                        + (1 - _lambda) * (N_nr / N) * loss_non_rare
                    )

                loss = loss / self.cfg.model.gradient_accumulate_every
                self.optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()

                if (i + 1) % self.cfg.model.gradient_accumulate_every == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.ema.update()

            self.scheduler.step(total_loss)

            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=self.current_epoch)

        print("Training complete")

    def load(self, path: str):
        checkpoint = torch.load(path)

        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded regular model state.")
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")

        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print(
                "No optimizer state found in checkpoint or optimizer not initialized."
            )

        if "ema_state_dict" in checkpoint and hasattr(self, "ema"):
            self.ema.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            print("Loaded EMA model state.")
        else:
            print("No EMA state found in checkpoint or EMA not initialized.")

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

        self.to(self.device)
        if hasattr(self, "ema") and self.ema.ema_model:
            self.ema.ema_model.to(self.device)
        print(f"Model and EMA model moved to {self.device}.")

    def save(self, path: str = None, epoch: int = None):

        if path is None:
            hydra_output_dir = os.path.join(self.cfg.run_dir)

            if not os.path.exists(os.path.join(hydra_output_dir, "checkpoints")):
                os.makedirs(
                    os.path.join(hydra_output_dir, "checkpoints"), exist_ok=True
                )

            path = os.path.join(
                os.path.join(hydra_output_dir, "checkpoints"),
                f"diffusion_ts_checkpoint_{epoch if epoch else self.current_epoch}.pt",
            )

        model_state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}
        optimizer_state_dict_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self.optimizer.state_dict().items()
        }
        ema_state_dict_cpu = {
            k: v.cpu() for k, v in self.ema.ema_model.state_dict().items()
        }
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict_cpu,
                "optimizer_state_dict": optimizer_state_dict_cpu,
                "ema_state_dict": ema_state_dict_cpu,
                "conditioning_module_state_dict": self.conditioning_module.state_dict(),
            },
            path,
        )


class EMA:
    def __init__(self, model, beta, update_every, device):
        super(EMA, self).__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model).eval().to(device)
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        self.device = device
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        self.step += 1
        if self.step % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.beta).add_(
                    model_param.data, alpha=1.0 - self.beta
                )

    def forward(self, x):
        return self.ema_model(x)
