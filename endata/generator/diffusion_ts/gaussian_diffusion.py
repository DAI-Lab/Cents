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
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from endata.generator.base_generator import BaseGenerator
from endata.generator.context import ContextModule
from endata.generator.diffusion_ts.model_utils import default, extract, identity
from endata.generator.diffusion_ts.transformer import Transformer


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


class EMA:
    """
    Exponential Moving Average to stabilize training.
    """

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


class Diffusion_TS(nn.Module, BaseGenerator):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.eta = cfg.model.eta
        self.use_ff = cfg.model.use_ff
        self.seq_len = cfg.dataset.seq_len
        self.time_series_dims = cfg.dataset.time_series_dims
        self.ff_weight = default(cfg.model.reg_weight, math.sqrt(self.seq_len) / 5)
        self.device = cfg.device

        self.cond_loss_weight = cfg.model.cond_loss_weight

        self.embedding_dim = cfg.model.cond_emb_dim
        self.context_var_n_categories = cfg.dataset.context_vars
        self.context_module = ContextModule(
            self.context_var_n_categories, self.embedding_dim, self.device
        ).to(self.device)

        self.fc = nn.Linear(
            self.time_series_dims + self.embedding_dim, self.time_series_dims
        )
        self.model = Transformer(
            n_feat=self.time_series_dims + self.embedding_dim,
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

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        regbuf("posterior_variance", posterior_variance)
        regbuf(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        pmc1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        pmc2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        regbuf("posterior_mean_coef1", pmc1)
        regbuf("posterior_mean_coef2", pmc2)

        lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100
        regbuf("loss_weight", lw)

        if self.loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.recon_loss_fn = F.mse_loss
        else:
            raise ValueError(f"Invalid loss type {self.loss_type}")

        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.wandb_enabled = getattr(self.cfg, "wandb_enabled", False)

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

    def forward(self, x, context_vars=None):
        """
        Single forward step for a random diffusion time t:
          1) Sample t in [0, self.num_timesteps)
          2) Inject noise into x according to q_sample
          3) Forward through Transformer to predict denoised x
          4) Return the reconstruction loss + classification_logits from cond. module
        """
        b, s, d = x.shape
        assert s == self.seq_len and d == self.time_series_dims

        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

        # 2) Condition module forward => (embedding, classification_logits)
        embedding, cond_classification_logits = self.context_module(context_vars)

        # 3) Add noise to x (q_sample)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise=noise)

        # 4) Concat embedding to x_noisy, feed through transformer, map with self.fc
        #    shape: [b, s, d + embed_dim]
        c = torch.cat([x_noisy, embedding.unsqueeze(1).repeat(1, s, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        model_out = trend + season
        x_recon = self.fc(model_out)

        # 5) Compute reconstruction loss (weighted by self.loss_weight[t])
        rec_loss = self.recon_loss_fn(x_recon, x, reduction="none")  # shape: [b, s, d]
        if self.use_ff:
            # optional frequency-domain reg
            fft_pred = torch.fft.fft(x_recon.transpose(1, 2), norm="forward")
            fft_true = torch.fft.fft(x.transpose(1, 2), norm="forward")
            fft_pred, fft_true = fft_pred.transpose(1, 2), fft_true.transpose(1, 2)
            fl = self.recon_loss_fn(
                torch.real(fft_pred), torch.real(fft_true), reduction="none"
            )
            fl += self.recon_loss_fn(
                torch.imag(fft_pred), torch.imag(fft_true), reduction="none"
            )
            rec_loss += self.ff_weight * fl

        rec_loss = reduce(rec_loss, "b ... -> b (...)", "mean")
        lw = extract(self.loss_weight, t, rec_loss.shape)
        rec_loss = (rec_loss * lw).mean()

        return rec_loss, cond_classification_logits

    def q_sample(self, x_start, t, noise=None):
        n = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * n
        )

    def model_predictions(self, x, t, embedding, clip_x_start=False):
        """
        Given current x, time t, and the context embedding,
        produce x_start (denoised) and pred_noise.
        """
        c = torch.cat([x, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        model_out = trend + season
        x_start = self.fc(model_out)
        if clip_x_start:
            x_start = x_start.clamp(-1.0, 1.0)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, embedding, clip_denoised=True):
        pred_noise, x_start = self.model_predictions(
            x, t, embedding, clip_x_start=clip_denoised
        )
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def p_sample(self, x, t, embedding, clip_denoised=True):
        bt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pm, pv, plv, x_start = self.p_mean_variance(x, bt, embedding, clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.0
        return pm + (0.5 * plv).exp() * noise, x_start

    @torch.no_grad()
    def sample(self, shape, context_vars):
        """
        Ancestral sampling from random noise down to x0.
        """
        dev = self.betas.device
        img = torch.randn(shape, device=dev)
        embedding, _ = self.context_module(context_vars)

        for t in tqdm(
            reversed(range(self.num_timesteps)),
            desc="Sampling",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t, embedding)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, context_vars, clip_denoised=True):
        """
        DDIM-like fast sampling approach.
        """
        dev = self.betas.device
        batch = shape[0]
        # get embedding once
        with torch.no_grad():
            embedding, _ = self.context_module(context_vars)

        tt = self.num_timesteps
        st = self.sampling_timesteps
        eta = self.eta
        times = torch.linspace(-1, tt - 1, steps=st + 1)
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=dev)
        for time, time_next in tqdm(pairs, desc="Fast Sampling"):
            bt = torch.full((batch,), time, device=dev, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                img, bt, embedding, clip_x_start=clip_denoised
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

    def train_model(self, train_dataset):
        self.train()
        self.to(self.device)

        loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.model.batch_size,
            shuffle=self.cfg.dataset.shuffle,
            drop_last=True,
        )

        self.optimizer = Adam(
            self.parameters(), lr=self.cfg.model.base_lr, betas=[0.9, 0.96]
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

        num_epochs = self.cfg.model.n_epochs
        use_fp16 = getattr(self.cfg.model, "use_fp16", False)
        scaler = None
        if use_fp16:
            from torch.cuda.amp import GradScaler, autocast

            scaler = GradScaler()

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for ts_batch, cond_batch in loader:
                ts_batch = ts_batch.to(self.device)
                for k in cond_batch:
                    cond_batch[k] = cond_batch[k].to(self.device)

                if use_fp16:
                    with torch.cuda.amp.autocast():
                        rec_loss, cond_class_logits = self(ts_batch, cond_batch)
                        cond_loss = 0.0
                        for var_name, logdits in cond_class_logits.items():
                            labels = cond_batch[var_name]
                            cond_loss += self.auxiliary_loss(logits, labels)
                        total_loss = rec_loss + self.cond_loss_weight * cond_loss
                else:
                    rec_loss, cond_class_logits = self(ts_batch, cond_batch)
                    cond_loss = 0.0
                    for var_name, logits in cond_class_logits.items():
                        labels = cond_batch[var_name]
                        cond_loss += self.auxiliary_loss(logits, labels)
                    total_loss = rec_loss + self.cond_loss_weight * cond_loss

                self.optimizer.zero_grad()

                if use_fp16:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()

                self.ema.update()
                epoch_loss += total_loss.item()

            self.scheduler.step(epoch_loss)

            if self.wandb_enabled and wandb is not None:
                wandb.log(
                    {
                        "Loss/Total": epoch_loss,
                        "Loss/Recon": rec_loss.item(),
                        "Loss/Cond": cond_loss.item(),
                        "Epoch": epoch + 1,
                    }
                )

            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=epoch + 1)

        print("Training complete")

    def save(self, path: str = None, epoch: int = None):
        if path is None:
            run_dir = os.path.join(self.cfg.run_dir)
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(
                ckpt_dir,
                f"diffusion_ts_checkpoint_{epoch if epoch else 0}.pt",
            )

        m_sd = {k: v.cpu() for k, v in self.state_dict().items()}
        opt_sd = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in self.optimizer.state_dict().items()
        }
        ema_sd = {k: v.cpu() for k, v in self.ema.ema_model.state_dict().items()}

        torch.save(
            {
                "epoch": epoch if epoch is not None else 0,
                "model_state_dict": m_sd,
                "optimizer_state_dict": opt_sd,
                "ema_state_dict": ema_sd,
                "context_module_state_dict": self.context_module.state_dict(),
            },
            path,
        )
        print(f"Saved diffusion model checkpoint to {path}")

    def load(self, path: str):

        if isinstance(self.device, str):
            if self.device == "cpu":
                map_location = torch.device("cpu")
            else:
                raise ValueError(f"Invalid device string: {self.device}")
        elif isinstance(self.device, int):
            if (
                torch.cuda.is_available()
                and 0 <= self.device < torch.cuda.device_count()
            ):
                map_location = torch.device(f"cuda:{self.device}")
            else:
                raise ValueError(f"Invalid CUDA device index: {self.device}")
        else:
            raise TypeError(
                f"Device should be a string or an integer, but got {type(self.device)}"
            )

        ckp = torch.load(path, map_location=map_location)

        if "model_state_dict" in ckp:
            self.load_state_dict(ckp["model_state_dict"])
            print("Loaded model state.")
        else:
            raise KeyError("Checkpoint missing 'model_state_dict'.")

        if "optimizer_state_dict" in ckp and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print("No optimizer state or not initialized.")

        if "ema_state_dict" in ckp and hasattr(self, "ema"):
            self.ema.ema_model.load_state_dict(ckp["ema_state_dict"])
            print("Loaded EMA model state.")
        else:
            print("No EMA state found or not initialized.")

        if "context_module_state_dict" in ckp:
            self.context_module.load_state_dict(ckp["context_module_state_dict"])
            print("Loaded context module state.")
        else:
            print("No context module state found in checkpoint.")

        if "epoch" in ckp:
            print(f"Loaded epoch number: {ckp['epoch']}")

        self.to(self.device)
        print(f"Model + EMA moved to {self.device}.")
