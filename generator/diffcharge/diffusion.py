import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm.auto import tqdm

from datasets.utils import prepare_dataloader
from generator.conditioning import ConditioningModule
from generator.diffcharge.network import CNN, Attention

try:
    import wandb
except ImportError:
    wandb = None


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
    Exponential Moving Average for model parameters.
    """

    def __init__(self, model, beta, update_every, device):
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


class DDPM(nn.Module):
    """
    A simplified DDPM that:
      - Uses new ConditioningModule returning (embedding, classification_logits).
      - Includes classification loss for each context variable, weighted by cond_loss_weight.
      - Removes warm-up, KL, GMM, etc.
      - Maintains the original p_sample / sampling approach.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.n_steps = cfg.model.n_steps
        self.schedule = cfg.model.schedule
        self.beta_start = cfg.model.beta_start
        self.beta_end = cfg.model.beta_end
        self.current_epoch = 0
        self.context_var_n_categories = cfg.dataset.conditioning_vars
        self.cond_loss_weight = cfg.model.cond_loss_weight

        self.conditioning_module = ConditioningModule(
            context_vars=self.context_var_n_categories,
            embedding_dim=cfg.model.cond_emb_dim,
            device=self.device,
        )

        if cfg.model.network == "attention":
            self.eps_model = Attention(cfg).to(self.device)
        else:
            self.eps_model = CNN(cfg).to(self.device)

        if self.schedule == "linear":
            self.beta = linear_beta_schedule(self.n_steps, self.device)
        elif self.schedule == "cosine":
            self.beta = cosine_beta_schedule(self.n_steps, self.device)
        else:
            self.beta = (
                torch.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.n_steps,
                    device=self.device,
                )
                ** 2
            )

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sigma2 = torch.cat(
            (
                torch.tensor([self.beta[0]], device=self.device),
                self.beta[1:] * (1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:]),
            )
        )

        self.loss_func = nn.MSELoss()
        self.aux_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.eps_model.parameters())
            + list(self.conditioning_module.parameters()),
            lr=cfg.model.init_lr,
        )

        n_epochs = cfg.model.n_epochs
        p1, p2 = int(0.75 * n_epochs), int(0.9 * n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

        self.ema = EMA(
            self.eps_model,
            beta=cfg.model.ema_decay,
            update_every=cfg.model.ema_update_interval,
            device=self.device,
        )

        self.wandb_enabled = getattr(self.cfg, "wandb_enabled", False)
        if self.wandb_enabled and wandb is not None:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=cfg,
                dir=cfg.run_dir,
            )

    def gather(self, const, t):
        """
        Simple gather for 1D buffer [n_steps], indexing by (B, ) shaped t.
        Returns shape (B, 1, 1).
        """
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = alpha_bar.sqrt() * x0
        var = 1.0 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        """
        Forward process: x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
        """
        mean, var = self.q_xt_x0(x0, t)
        return mean + var.sqrt() * eps

    def p_sample_step(self, x_t, embedding, t):
        """
        Single step of reverse diffusion:
          x_{t-1} = (x_t - (1 - alpha[t]) / sqrt(1 - alpha_bar[t]) * eps_theta) / sqrt(alpha[t])
                     + sqrt(sigma2[t]) * noise
        """
        B, seq_len, d_in = x_t.shape
        emb_rep = embedding.unsqueeze(1).repeat(1, seq_len, 1)
        inp = torch.cat([x_t, emb_rep], dim=-1)  # feed to eps_model

        eps_theta = self.eps_model(inp, t)

        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()

        mean = (x_t - eps_coef * eps_theta) / alpha.sqrt()

        var = self.gather(self.sigma2, t)
        noise = torch.zeros_like(x_t) if (t == 0).all() else torch.randn_like(x_t)
        return mean + var.sqrt() * noise

    def diffusion_loss(self, x0, embedding):
        """
        Basic DDPM training loss:
          1) Sample random t
          2) Add noise
          3) Predict noise
          4) Compare predicted noise vs. actual noise with MSE
        """
        bsz = x0.shape[0]
        t = torch.randint(0, self.n_steps, (bsz,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Concat x_t + embedding => predict noise
        B, seq_len, _ = x_t.shape
        emb_rep = embedding.unsqueeze(1).repeat(1, seq_len, 1)
        eps_theta = self.eps_model(torch.cat([x_t, emb_rep], dim=-1), t)
        return self.loss_func(noise, eps_theta)

    def train_model(self, train_dataset):
        """
        Training loop that includes classification loss from the conditioning module.
        """
        self.train()
        self.to(self.device)

        train_loader = prepare_dataloader(
            train_dataset, batch_size=self.cfg.model.batch_size, shuffle=True
        )
        n_epochs = self.cfg.model.n_epochs

        for epoch in range(n_epochs):
            self.current_epoch = epoch + 1
            batch_losses = []
            for x0, cond_vars in tqdm(
                train_loader, desc=f"Epoch {epoch+1}", leave=False
            ):
                x0 = x0.to(self.device)
                for k in cond_vars:
                    cond_vars[k] = cond_vars[k].to(self.device)

                embedding, classification_logits = self.conditioning_module(cond_vars)
                loss_main = self.diffusion_loss(x0, embedding)

                cond_loss = 0.0
                for var_name, logits in classification_logits.items():
                    labels = cond_vars[var_name]
                    cond_loss += self.aux_loss(logits, labels)

                # 4) Total
                total_loss = loss_main + self.cond_loss_weight * cond_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.ema.update()

                batch_losses.append(total_loss.item())

                if self.wandb_enabled and wandb is not None:
                    wandb.log(
                        {
                            "Loss/Diffusion": loss_main.item(),
                            "Loss/CondClassification": cond_loss.item(),
                            "Loss/Total": total_loss.item(),
                        }
                    )

            epoch_loss = sum(batch_losses) / len(batch_losses)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
            self.lr_scheduler.step(epoch_loss)

        print("Training complete")

    @torch.no_grad()
    def sample(self, shape, cond_vars, use_ema=False):
        """
        DDPM ancestral sampling.
        shape = (batch_size, seq_len, input_dim)
        """
        model = self.ema.ema_model if use_ema else self.eps_model
        device = self.device

        # Start from Gaussian noise
        x = torch.randn(shape, device=device)

        # Get final embedding from cond vars
        embedding, _ = self.conditioning_module(cond_vars)

        # Reverse diffusion from t = n_steps-1 down to 0
        for t_ in tqdm(reversed(range(self.n_steps)), desc="DDPM Sampling"):
            t_tensor = torch.full((x.shape[0],), t_, device=device, dtype=torch.long)
            B, seq_len, d_in = x.shape
            emb_rep = embedding.unsqueeze(1).repeat(1, seq_len, 1)
            eps_theta = model(torch.cat([x, emb_rep], dim=-1), t_tensor)

            alpha_bar = self.gather(self.alpha_bar, t_tensor)
            alpha = self.gather(self.alpha, t_tensor)
            eps_coef = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
            mean = (x - eps_coef * eps_theta) / alpha.sqrt()

            var = self.gather(self.sigma2, t_tensor)
            noise = torch.zeros_like(x) if (t_ == 0) else torch.randn_like(x)
            x = mean + var.sqrt() * noise

        return x

    def generate(self, conditioning_vars):
        """
        Public method to generate multiple samples.
        """
        bs = self.cfg.model.sampling_batch_size
        seq_len = self.cfg.dataset.seq_len
        input_dim = self.cfg.dataset.input_dim
        total = len(next(iter(conditioning_vars.values())))
        generated_samples = []

        for start_idx in range(0, total, bs):
            end_idx = min(start_idx + bs, total)
            batch_conditioning_vars = {
                var_name: var_tensor[start_idx:end_idx]
                for var_name, var_tensor in conditioning_vars.items()
            }
            current_bs = end_idx - start_idx
            shape = (current_bs, seq_len, input_dim)

            with torch.no_grad():
                if getattr(self.cfg.model, "use_ema_sampling", False):
                    samples = self.sample(shape, batch_conditioning_vars, use_ema=True)
                else:
                    samples = self.sample(shape, batch_conditioning_vars, use_ema=False)

            generated_samples.append(samples)

        return torch.cat(generated_samples, dim=0)

    def save(self, path: str = None, epoch: int = None):
        if path is None:
            out_dir = os.path.join(self.cfg.run_dir, "checkpoints")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(
                out_dir, f"ddpm_checkpoint_{epoch or self.current_epoch}.pt"
            )

        model_sd = {k: v.cpu() for k, v in self.eps_model.state_dict().items()}
        opt_sd = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in self.optimizer.state_dict().items()
        }
        ema_sd = {k: v.cpu() for k, v in self.ema.ema_model.state_dict().items()}
        cond_sd = self.conditioning_module.state_dict()

        torch.save(
            {
                "epoch": epoch if epoch is not None else self.current_epoch,
                "eps_model_state_dict": model_sd,
                "optimizer_state_dict": opt_sd,
                "ema_state_dict": ema_sd,
                "alpha_bar": self.alpha_bar.cpu(),
                "beta": self.beta.cpu(),
                "conditioning_module_state_dict": cond_sd,
            },
            path,
        )
        print(f"DDPM checkpoint saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        ckp = torch.load(path, map_location=self.device)
        if "eps_model_state_dict" in ckp:
            self.eps_model.load_state_dict(ckp["eps_model_state_dict"])
            print("Loaded eps_model state.")
        if "optimizer_state_dict" in ckp:
            self.optimizer.load_state_dict(ckp["optimizer_state_dict"])
            print("Loaded optimizer state.")
        if "ema_state_dict" in ckp:
            self.ema.ema_model.load_state_dict(ckp["ema_state_dict"])
            print("Loaded EMA model state.")
        if "alpha_bar" in ckp:
            self.alpha_bar = ckp["alpha_bar"].to(self.device)
            print("Loaded alpha_bar.")
        if "beta" in ckp:
            self.beta = ckp["beta"].to(self.device)
            print("Loaded beta.")
        if "conditioning_module_state_dict" in ckp:
            self.conditioning_module.load_state_dict(
                ckp["conditioning_module_state_dict"]
            )
            print("Loaded conditioning module state.")
        if "epoch" in ckp:
            self.current_epoch = ckp["epoch"]
            print(f"Loaded epoch number: {self.current_epoch}")
        self.eps_model.to(self.device)
        self.conditioning_module.to(self.device)
        self.ema.ema_model.to(self.device)
        print(f"DDPM loaded and moved to {self.device}.")
