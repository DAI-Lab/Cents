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
from functools import partial

import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from generator.diffusion_ts.model_utils import default, extract, identity
from generator.diffusion_ts.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(
        device
    )


def cosine_beta_schedule(timesteps, s=0.004):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32).to(device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to(device)


class Diffusion_TS(nn.Module):
    def __init__(self, opt):
        super(Diffusion_TS, self).__init__()
        self.opt = opt
        self.eta, self.use_ff = opt.eta, opt.use_ff
        self.seq_len = opt.seq_len
        self.input_dim = opt.input_dim
        self.ff_weight = default(opt.reg_weight, math.sqrt(opt.seq_len) / 5)

        # Embedding layers for month and weekday
        self.month_embed = nn.Embedding(12, opt.cond_emb_dim)
        self.weekday_embed = nn.Embedding(7, opt.cond_emb_dim)

        self.fc = nn.Linear(opt.input_dim + opt.d_model * 2, opt.input_dim)

        self.model = Transformer(
            n_feat=opt.input_dim + opt.d_model * 2,
            n_channel=opt.seq_len,
            n_layer_enc=opt.n_layer_enc,
            n_layer_dec=opt.n_layer_dec,
            n_heads=opt.n_heads,
            attn_pdrop=opt.attn_pd,
            resid_pdrop=opt.resid_pd,
            mlp_hidden_times=opt.mlp_hidden_times,
            max_len=opt.seq_len,
            n_embd=opt.d_model,
            conv_params=[opt.kernel_size, opt.padding_size],
        )

        if opt.beta_schedule == "linear":
            betas = linear_beta_schedule(opt.n_steps)
        elif opt.beta_schedule == "cosine":
            betas = cosine_beta_schedule(opt.n_steps)
        else:
            raise ValueError(f"unknown beta schedule {opt.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = opt.loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            opt.sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

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

        # calculate reweighting

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

    def output(self, x, t, padding_masks=None, weekday=None, month=None):
        if month is not None and weekday is not None:
            month_emb = (
                self.month_embed(month)
                .unsqueeze(1)
                .repeat(1, self.seq_len, 1)
                .to(device)
            )
            weekday_emb = (
                self.weekday_embed(weekday)
                .unsqueeze(1)
                .repeat(1, self.seq_len, 1)
                .to(device)
            )
            x = torch.cat([x, month_emb, weekday_emb], dim=-1)
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return self.fc(model_output)

    @torch.no_grad()
    def sample(self, shape, labels):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t, labels)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, labels, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, labels, clip_x_start=clip_denoised
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

    def add_conditioning(self, x, month, weekday):
        month_emb = self.month_embed(month).unsqueeze(1).repeat(1, self.seq_len, 1)
        weekday_emb = (
            self.weekday_embed(weekday).unsqueeze(1).repeat(1, self.seq_len, 1)
        )
        return torch.cat([x, month_emb, weekday_emb], dim=-1)

    def model_predictions(self, x, t, labels, clip_x_start=False):
        weekday_labels, month_labels = labels
        conditioned_x = self.add_conditioning(x, month_labels, weekday_labels)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self.output(conditioned_x, t)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_sample(self, x, t: int, labels, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, labels=labels, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_mean_variance(self, x, t, labels, clip_denoised=True):
        _, x_start = self.model_predictions(x, t, labels)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def generate_mts(self, batch_size=16):
        input_dim, seq_len = self.self.input_dim, self.seq_len
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn((batch_size, seq_len, input_dim))

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
        weekday=None,
        month=None,
    ):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks, weekday=weekday, month=month)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        fourier_loss = torch.tensor([0.0]).to(device)
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
        return train_loss.mean()

    def forward(self, x, weekday=None, month=None, **kwargs):
        (
            b,
            c,
            n,
            device,
            input_dim,
        ) = (
            *x.shape,
            x.device,
            self.input_dim,
        )
        assert n == input_dim, f"number of variable must be {input_dim}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, month=month, weekday=weekday, **kwargs)

    def return_components(self, x, t: int):
        (
            b,
            c,
            n,
            device,
            input_dim,
        ) = (
            *x.shape,
            x.device,
            self.input_dim,
        )
        assert n == input_dim, f"number of variable must be {input_dim}"
        t = torch.tensor([t]).to(device)
        t = t.repeat(b)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def train_model(self, train_dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=self.opt.shuffle
        )

        # Create necessary directories
        os.makedirs(self.opt.results_folder, exist_ok=True)

        # Optimizer and EMA setup
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.opt.base_lr,
            betas=[0.9, 0.96],
        )
        self.ema = EMA(
            self, beta=self.opt.ema_decay, update_every=self.opt.ema_update_interval
        ).to(device)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, **self.opt.lr_scheduler_params
        )

        # Training loop
        for epoch in tqdm(range(self.opt.n_epochs), desc="Training"):
            total_loss = 0.0
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                train_loader
            ):
                time_series_batch, month_label_batch, day_label_batch = (
                    time_series_batch.to(device),
                    month_label_batch.to(device),
                    day_label_batch.to(device),
                )
                loss = self(
                    time_series_batch, weekday=day_label_batch, month=month_label_batch
                )
                loss = loss / self.opt.gradient_accumulate_every
                loss.backward()
                total_loss += loss.item()

                if (i + 1) % self.opt.gradient_accumulate_every == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema.update()

            self.scheduler.step(total_loss)

            # Save model checkpoint
            if (epoch + 1) % self.opt.save_cycle == 0:
                checkpoint_path = os.path.join(
                    self.opt.results_folder, f"checkpoint-{epoch + 1}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "ema_state_dict": self.ema.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint at {checkpoint_path}")

        print("Training complete")

    def generate(self, day_labels, month_labels):
        num_samples = day_labels.shape[0]
        shape = (num_samples, self.seq_len, self.input_dim)
        return self._generate(shape, [day_labels, month_labels])

    def _generate(self, shape, labels):
        self.eval()
        with torch.no_grad():
            samples = (
                self.fast_sample(shape, labels)
                if self.fast_sampling
                else self.sample(shape, labels)
            )
            return samples


class EMA(nn.Module):
    def __init__(self, model, beta, update_every):
        super(EMA, self).__init__()
        self.ema_model = copy.deepcopy(model).eval().to(device)
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
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.parameters()
            ):
                ema_param.data.mul_(self.beta).add_(
                    model_param.data, alpha=1.0 - self.beta
                )

    def forward(self, x):
        return self.ema_model(x)


if __name__ == "__main__":
    pass
