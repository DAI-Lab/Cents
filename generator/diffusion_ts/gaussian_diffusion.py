"""
This class is adapted/taken from the Diffusion_TS GitHub repository:

Repository: https://github.com/Y-debug-sys/Diffusion-TS
Author: Xinyu Yuan
License: MIT License

Modifications:
- Conditioning and sampling logic
- Added further functions and removed unused functionality
- Warm-up logic for the conditioning module
- GMM-based rarity detection after warm-up

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import copy
import math
import os
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
import wandb
from einops import reduce
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture
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
        super().__init__()
        self.cfg = cfg
        self.eta = cfg.model.eta
        self.use_ff = cfg.model.use_ff
        self.seq_len = cfg.dataset.seq_len
        self.input_dim = cfg.dataset.input_dim
        self.ff_weight = default(cfg.model.reg_weight, math.sqrt(self.seq_len) / 5)
        self.device = cfg.device
        self.embedding_dim = cfg.model.cond_emb_dim
        self.conditioning_var_n_categories = cfg.dataset.conditioning_vars
        self.warm_up_epochs = cfg.model.warm_up_epochs
        self.sparse_conditioning_loss_weight = cfg.model.sparse_conditioning_loss_weight
        self.kl_weight = cfg.model.kl_weight
        self.gmm_fitted = False
        self.current_epoch = 0
        self.wandb_enabled = getattr(self.cfg, "wandb_enabled", False)

        self.conditioning_module = ConditioningModule(
            self.conditioning_var_n_categories, self.embedding_dim, self.device
        ).to(self.device)

        self.fc = nn.Linear(self.input_dim + self.embedding_dim, self.input_dim)

        self.model = Transformer(
            n_feat=self.input_dim + self.embedding_dim,
            n_channel=cfg.dataset.seq_len,
            n_layer_enc=cfg.model.n_layer_enc,
            n_layer_dec=cfg.model.n_layer_dec,
            n_heads=cfg.model.n_heads,
            attn_pdrop=cfg.model.attn_pd,
            resid_pdrop=cfg.model.resid_pd,
            mlp_hidden_times=cfg.model.mlp_hidden_times,
            max_len=cfg.dataset.seq_len,
            n_embd=cfg.model.d_model,
            conv_params=[cfg.model.kernel_size, cfg.model.padding_size],
        )

        if cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.model.n_steps, self.device)
        elif cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.model.n_steps, self.device)
        else:
            raise ValueError(f"unknown beta schedule {cfg.model.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(
            self.device
        )

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = cfg.model.loss_type
        self.sampling_timesteps = default(cfg.model.sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

        def regbuf(name, val):
            self.register_buffer(name, val.to(torch.float32))

        regbuf("betas", betas)
        regbuf("alphas_cumprod", alphas_cumprod)
        regbuf("alphas_cumprod_prev", alphas_cumprod_prev)
        regbuf("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        regbuf("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        regbuf("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        regbuf("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        regbuf("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        pv = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        regbuf("posterior_variance", pv)
        regbuf("posterior_log_variance_clipped", torch.log(pv.clamp(min=1e-20)))
        pmc1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        pmc2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        regbuf("posterior_mean_coef1", pmc1)
        regbuf("posterior_mean_coef2", pmc2)
        lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100
        regbuf("loss_weight", lw)

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
        pm = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        pv = extract(self.posterior_variance, t, x_t.shape)
        plv = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return pm, pv, plv

    def output(self, x, t, padding_masks=None, conditioning_vars=None):
        mu, logvar = None, None
        if conditioning_vars is not None:
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=False)
            cvec = mu.unsqueeze(1).repeat(1, self.seq_len, 1)
            x = torch.cat([x, cvec], dim=-1)
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_out = trend + season
        out = self.fc(model_out)
        return out, mu, logvar

    def model_predictions(self, x, t, conditioning_vars, clip_x_start=False):
        fn = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        x_start, mu, logvar = self.output(x, t, conditioning_vars=conditioning_vars)
        x_start = fn(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_sample(self, x, t, conditioning_vars, clip_denoised=True):
        bt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        mm, _, mlv, x_start = self.p_mean_variance(
            x, bt, conditioning_vars, clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        return mm + (0.5 * mlv).exp() * noise, x_start

    def p_mean_variance(self, x, t, conditioning_vars, clip_denoised=True):
        pred_noise, x_start = self.model_predictions(
            x, t, conditioning_vars, clip_x_start=clip_denoised
        )
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def sample(self, shape, conditioning_vars):
        dev = self.betas.device
        img = torch.randn(shape, device=dev)
        loop = tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        )
        for t in loop:
            img, _ = self.p_sample(img, t, conditioning_vars)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, conditioning_vars, clip_denoised=True):
        batch = shape[0]
        dev = self.betas.device
        tt = self.num_timesteps
        st = self.sampling_timesteps
        eta = self.eta
        times = torch.linspace(-1, tt - 1, steps=st + 1)
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=dev)
        loop = tqdm(pairs, desc="sampling loop time step")
        for time, time_next in loop:
            bt = torch.full((batch,), time, device=dev, dtype=torch.long)
            pn, xst = self.model_predictions(
                img, bt, conditioning_vars, clip_x_start=clip_denoised
            )
            if time_next < 0:
                img = xst
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = xst * alpha_next.sqrt() + c * pn + sigma * noise
        return img

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        d = {}
        if random:
            for v, nc in self.conditioning_var_n_categories.items():
                d[v] = torch.randint(
                    0, nc, (batch_size,), dtype=torch.long, device=self.device
                )
        else:
            rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for v in self.conditioning_var_n_categories.keys():
                d[v] = torch.tensor(
                    rows[v].values, dtype=torch.long, device=self.device
                )
        return d

    def _generate(self, shape, conditioning_vars):
        self.eval()
        with torch.no_grad():
            return (
                self.fast_sample(shape, conditioning_vars)
                if self.fast_sampling
                else self.sample(shape, conditioning_vars)
            )

    def generate(self, conditioning_vars):
        bs = self.cfg.model.sampling_batch_size
        total = len(next(iter(conditioning_vars.values())))
        generated_samples = []

        for start_idx in range(0, total, bs):
            end_idx = min(start_idx + bs, total)
            batch_conditioning_vars = {
                var_name: var_tensor[start_idx:end_idx]
                for var_name, var_tensor in conditioning_vars.items()
            }
            current_bs = end_idx - start_idx
            shape = (current_bs, self.seq_len, self.input_dim)

            if getattr(self.cfg.model, "use_ema_sampling", False):
                samples = self.ema.ema_model._generate(shape, batch_conditioning_vars)
            else:
                samples = self._generate(shape, batch_conditioning_vars)

            generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def q_sample(self, x_start, t, noise=None):
        n = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * n
        )

    def _train_loss(self, x_start, t, target=None, noise=None, conditioning_vars=None):
        with torch.set_grad_enabled(self.training):
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=False)
        n = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start
        x = self.q_sample(x_start, t, noise=n)
        c = torch.cat([x, mu.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        mo = self.fc(trend + season)
        loss_main = self.loss_fn(mo, target, reduction="none")
        if self.use_ff:
            fft1 = torch.fft.fft(mo.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fl = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            loss_main += self.ff_weight * fl
        loss_main = reduce(loss_main, "b ... -> b (...)", "mean")
        lw = extract(self.loss_weight, t, loss_main.shape)
        loss_main = loss_main * lw
        return loss_main.mean(), mu, logvar

    def forward(self, x, conditioning_vars=None, **kwargs):
        b, s, d = x.shape
        assert s == self.seq_len
        assert d == self.input_dim
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        loss, mu, logvar = self._train_loss(
            x, t, conditioning_vars=conditioning_vars, **kwargs
        )
        return loss, mu, logvar

    def _fit_gmm(self, loader):
        all_mu = []
        self.conditioning_module.eval()
        with torch.no_grad():
            for bx, cond_vars in loader:
                bx = bx.to(self.device)
                for k in cond_vars:
                    cond_vars[k] = cond_vars[k].to(self.device)
                l, mu, _ = self(bx, cond_vars)
                all_mu.append(mu.cpu())
        a = torch.cat(all_mu, dim=0)
        self.conditioning_module.fit_gmm(a)
        self.conditioning_module.set_rare_threshold(a, fraction=0.1)
        self.gmm_fitted = True

    def train_model(self, train_dataset):
        self.train_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.train()
        self.to(self.device)
        loader = DataLoader(
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
        if self.wandb_enabled and wandb is not None:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=self.cfg,
                dir=self.cfg.run_dir,
            )

        for epoch in tqdm(
            range(self.cfg.model.n_epochs), desc="Epoch", total=self.cfg.model.n_epochs
        ):
            self.current_epoch = epoch + 1
            total_loss = 0.0
            for param in self.conditioning_module.parameters():
                param.requires_grad = True
            if self.current_epoch > self.warm_up_epochs:
                for param in self.conditioning_module.parameters():
                    param.requires_grad = False
            for i, (ts_batch, cond_batch) in enumerate(loader):
                ts_batch = ts_batch.to(self.device)
                for k in cond_batch:
                    cond_batch[k] = cond_batch[k].to(self.device)
                bsz = ts_batch.size(0)
                if self.current_epoch <= self.warm_up_epochs:
                    loss, mu, logvar = self(ts_batch, conditioning_vars=cond_batch)
                    kl_loss_val = 0.0
                    if mu is not None and logvar is not None:
                        kl_loss_t = self.conditioning_module.kl_divergence(mu, logvar)
                        kl_loss_val = kl_loss_t.item()
                        loss = loss + self.kl_weight * kl_loss_t
                    if self.wandb_enabled and wandb is not None:
                        wandb.log(
                            {"Loss/reconstruction": loss.item(), "Loss/KL": kl_loss_val}
                        )
                else:
                    with torch.no_grad():
                        _, mu, _ = self(ts_batch, conditioning_vars=cond_batch)
                        md = mu.detach()
                        if self.gmm_fitted:
                            rm = (
                                self.conditioning_module.is_rare(md)
                                .float()
                                .to(self.device)
                            )
                        else:
                            rm = torch.zeros((bsz,), device=self.device)
                    ri = (rm == 1.0).nonzero(as_tuple=True)[0]
                    nr = (rm == 0.0).nonzero(as_tuple=True)[0]
                    loss_rare = torch.tensor(0.0, device=self.device)
                    loss_non_rare = torch.tensor(0.0, device=self.device)
                    if len(ri) > 0:
                        ts_r = ts_batch[ri]
                        c_r = {kk: vv[ri] for kk, vv in cond_batch.items()}
                        loss_r, _, _ = self(ts_r, c_r)
                        loss_rare = loss_r
                    if len(nr) > 0:
                        ts_nr = ts_batch[nr]
                        c_nr = {kk: vv[nr] for kk, vv in cond_batch.items()}
                        loss_nr, _, _ = self(ts_nr, c_nr)
                        loss_non_rare = loss_nr
                    n_r = rm.sum().item()
                    n_nr = (1 - rm).sum().item()
                    n_tot = float(bsz)
                    lam = self.sparse_conditioning_loss_weight
                    loss = (
                        lam * (n_r / n_tot) * loss_rare
                        + (1 - lam) * (n_nr / n_tot) * loss_non_rare
                    )
                    if self.wandb_enabled and wandb is not None:
                        wandb.log({"Loss/reconstruction": loss.item()})
                loss = loss / self.cfg.model.gradient_accumulate_every
                self.optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()
                if (i + 1) % self.cfg.model.gradient_accumulate_every == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.ema.update()
            self.scheduler.step(total_loss)
            if self.current_epoch == self.warm_up_epochs and not self.gmm_fitted:
                self._fit_gmm(loader)
            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=self.current_epoch)
        print("Training complete")

    def load(self, path: str):
        ckp = torch.load(path)
        if "model_state_dict" in ckp:
            self.load_state_dict(ckp["model_state_dict"])
            print("Loaded regular model state.")
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")
        if "optimizer_state_dict" in ckp and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print(
                "No optimizer state found in checkpoint or optimizer not initialized."
            )
        if "ema_state_dict" in ckp and hasattr(self, "ema"):
            self.ema.ema_model.load_state_dict(ckp["ema_state_dict"])
            print("Loaded EMA model state.")
        else:
            print("No EMA state found in checkpoint or EMA not initialized.")
        if "conditioning_module_state_dict" in ckp:
            self.conditioning_module.load_state_dict(
                ckp["conditioning_module_state_dict"]
            )
            print("Loaded conditioning module state.")
        else:
            print("No conditioning module state found in checkpoint.")
        if "epoch" in ckp:
            self.current_epoch = ckp["epoch"]
            print(f"Loaded epoch number: {self.current_epoch}")
        else:
            print("No epoch information found in checkpoint.")
        self.to(self.device)
        if hasattr(self, "ema") and self.ema.ema_model:
            self.ema.ema_model.to(self.device)
        print(f"Model and EMA model moved to {self.device}.")

    def save(self, path: str = None, epoch: int = None):
        if path is None:
            run_dir = os.path.join(self.cfg.run_dir)
            if not os.path.exists(os.path.join(run_dir, "checkpoints")):
                os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
            path = os.path.join(
                run_dir,
                "checkpoints",
                f"diffusion_ts_checkpoint_{epoch if epoch else self.current_epoch}.pt",
            )
        m_sd = {k: v.cpu() for k, v in self.state_dict().items()}
        opt_sd = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self.optimizer.state_dict().items()
        }
        ema_sd = {k: v.cpu() for k, v in self.ema.ema_model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": m_sd,
                "optimizer_state_dict": opt_sd,
                "ema_state_dict": ema_sd,
                "conditioning_module_state_dict": self.conditioning_module.state_dict(),
            },
            path,
        )


class EMA:
    def __init__(self, model, beta, update_every, device):
        super().__init__()
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
            for ema_p, model_p in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_p.data.mul_(self.beta).add_(model_p.data, alpha=1.0 - self.beta)

    def forward(self, x):
        return self.ema_model(x)
