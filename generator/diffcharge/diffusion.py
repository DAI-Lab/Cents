"""
This class is adapted/taken from the Diffusion_TS GitHub repository:

Repository: https://github.com/Y-debug-sys/Diffusion-TS
Author: Xinyu Yuan
License: MIT License

Modifications:
- Integrated conditioning logic using conditioning module.
- Simplified the conditioning handling by directly concatenating conditioning vectors.
- Maintained rare and non-rare sample handling for balanced training.
- Ensured compatibility with the modified model architecture.

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import copy
import math
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from datasets.utils import prepare_dataloader
from generator.conditioning import ConditioningModule
from generator.diffcharge.network import CNN, Attention
from generator.diffusion_ts.gaussian_diffusion import cosine_beta_schedule


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
    def __init__(self, cfg: DictConfig):
        super(DDPM, self).__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.conditioning_module = ConditioningModule(
            categorical_dims=cfg.dataset.conditioning_vars,
            embedding_dim=cfg.model.cond_emb_dim,
            device=cfg.device,
        ).to(self.device)

        if cfg.model.network == "attention":
            self.eps_model = Attention(cfg).to(self.device)
        else:
            self.eps_model = CNN(cfg).to(self.device)

        self.n_steps = cfg.model.n_steps
        schedule = cfg.model.schedule
        beta_start = cfg.model.beta_start
        beta_end = cfg.model.beta_end

        if schedule == "linear":
            self.beta = linear_beta_schedule(self.n_steps, self.device)
        elif schedule == "cosine":
            self.beta = cosine_beta_schedule(self.n_steps, self.device)
        else:
            self.beta = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
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
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=cfg.model.init_lr
        )
        self.loss_func = nn.MSELoss()
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

        self.current_epoch = 0
        self.sparse_conditioning_loss_weight = cfg.model.sparse_conditioning_loss_weight
        self.warm_up_epochs = cfg.model.warm_up_epochs

    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar.sqrt()) * x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + var.sqrt() * eps

    def p_sample_step(self, xt, z, t):
        eps_theta = self.eps_model(torch.cat([xt, z], dim=-1), t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean = (xt - eps_coef * eps_theta) / alpha.sqrt()
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z_noise = torch.zeros_like(xt)
        else:
            z_noise = torch.randn_like(xt)
        return mean + var.sqrt() * z_noise

    def cal_loss(self, x0, z, drop_prob=0.15):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)

        if torch.rand(1).item() < drop_prob:
            z = torch.zeros_like(z)

        eps_theta = self.eps_model(torch.cat([xt, z], dim=-1), t)
        return self.loss_func(noise, eps_theta)

    def train_model(self, train_dataset):
        self.train()
        self.to(self.device)

        train_loader = prepare_dataloader(
            train_dataset, self.cfg.model.batch_size, shuffle=True
        )

        for epoch in tqdm(range(self.cfg.model.n_epochs), desc="Training"):
            self.current_epoch = epoch + 1
            batch_loss = []

            if self.current_epoch > self.warm_up_epochs:
                for param in self.conditioning_module.parameters():
                    param.requires_grad = False

            for i, (time_series_batch, conditioning_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                x0 = time_series_batch.to(self.device)
                z, mu, logvar = self.conditioning_module(
                    conditioning_vars_batch, sample=False
                )

                if self.current_epoch > self.warm_up_epochs:
                    with torch.no_grad():
                        self.conditioning_module.update_running_statistics(mu.detach())
                        rare_mask = (
                            self.conditioning_module.is_rare(mu.detach())
                            .to(self.device)
                            .float()
                        )
                else:
                    self.conditioning_module.update_running_statistics(mu.detach())
                    rare_mask = torch.zeros((x0.size(0),), device=self.device)

                self.optimizer.zero_grad()

                if self.current_epoch <= self.warm_up_epochs:
                    # Before warm-up ends: treat all equally
                    loss = self.cal_loss(x0, z, drop_prob=0.1)
                    kl_loss = self.conditioning_module.kl_divergence(mu, logvar)
                    loss += self.kl_weight * kl_loss
                else:
                    rare_indices = (rare_mask == 1.0).nonzero(as_tuple=True)[0]
                    non_rare_indices = (rare_mask == 0.0).nonzero(as_tuple=True)[0]

                    loss_rare = torch.tensor(0.0, device=self.device)
                    loss_non_rare = torch.tensor(0.0, device=self.device)

                    if len(rare_indices) > 0:
                        x0_rare = x0[rare_indices]
                        z_rare = z[rare_indices]
                        loss_rare = self.cal_loss(x0_rare, z_rare, drop_prob=0.0)

                    if len(non_rare_indices) > 0:
                        x0_non_rare = x0[non_rare_indices]
                        z_non_rare = z[non_rare_indices]
                        loss_non_rare = self.cal_loss(
                            x0_non_rare, z_non_rare, drop_prob=0.0
                        )

                    N_r = rare_mask.sum().item()
                    N_nr = (rare_mask == 0.0).sum().item()
                    N = x0.size(0)
                    _lambda = self.sparse_conditioning_loss_weight

                    loss = (
                        _lambda * (N_r / N) * loss_rare
                        + (1 - _lambda) * (N_nr / N) * loss_non_rare
                    )

                loss.backward()
                self.optimizer.step()
                self.ema.update()

                batch_loss.append(loss.item())

            epoch_mean_loss = sum(batch_loss) / len(batch_loss)
            print(
                f"Epoch {epoch + 1}/{self.cfg.model.n_epochs}, Loss: {epoch_mean_loss:.4f}"
            )
            self.lr_scheduler.step(epoch_mean_loss)

            if (epoch + 1) % self.cfg.model.save_cycle == 0:
                self.save(epoch=self.current_epoch)

        print("Training complete")

    def save(self, path: str = None, epoch: int = None):
        if path is None:
            hydra_output_dir = os.path.join(self.cfg.run_dir)
            if not os.path.exists(os.path.join(hydra_output_dir, "checkpoints")):
                os.makedirs(
                    os.path.join(hydra_output_dir, "checkpoints"), exist_ok=True
                )
            path = os.path.join(
                os.path.join(hydra_output_dir, "checkpoints"),
                f"diffcharge_checkpoint_{epoch if epoch else self.current_epoch}.pt",
            )

        checkpoint = {
            "epoch": epoch if epoch is not None else self.current_epoch,
            "eps_model_state_dict": self.eps_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.ema_model.state_dict(),
            "alpha_bar": self.alpha_bar.cpu(),
            "beta": self.beta.cpu(),
            "conditioning_module_state_dict": self.conditioning_module.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Saved DDPM checkpoint to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)

        if "eps_model_state_dict" in checkpoint:
            self.eps_model.load_state_dict(checkpoint["eps_model_state_dict"])
            print("Loaded epsilon model state.")
        else:
            raise KeyError("Checkpoint does not contain 'eps_model_state_dict'.")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print("No optimizer state found in checkpoint.")

        if "ema_state_dict" in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            print("Loaded EMA model state.")
        else:
            print("No EMA state found in checkpoint.")

        if "alpha_bar" in checkpoint:
            self.alpha_bar = checkpoint["alpha_bar"].to(self.device)
            print("Loaded alpha_bar.")
        if "beta" in checkpoint:
            self.beta = checkpoint["beta"].to(self.device)
            print("Loaded beta.")

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

        self.eps_model.to(self.device)
        self.conditioning_module.to(self.device)
        self.ema.ema_model.to(self.device)
        print(f"DDPM models moved to {self.device}.")

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        conditioning_vars = {}
        if random:
            for var_name, num_categories in self.cfg.dataset.conditioning_vars.items():
                conditioning_vars[var_name] = torch.randint(
                    0,
                    num_categories,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.cfg.dataset.conditioning_vars.keys():
                conditioning_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, dtype=torch.long, device=self.device
                )

        return conditioning_vars

    def generate(self, conditioning_vars: dict, use_ema_sampling: bool = False):
        num_samples = conditioning_vars[next(iter(conditioning_vars))].shape[0]
        shape = (num_samples, self.cfg.dataset.seq_len, self.cfg.dataset.input_dim)

        with torch.no_grad():
            z, mu, logvar = self.conditioning_module(conditioning_vars, sample=True)

        if use_ema_sampling:
            print("Generating using EMA model parameters.")
            model = self.ema.ema_model
        else:
            print("Generating using regular epsilon model parameters.")
            model = self.eps_model

        samples, mu, logvar = self._generate(shape, z, mu, logvar, model=model)
        return samples

    @torch.no_grad()
    def _generate(self, shape, z, mu, logvar, model=None):
        if model is None:
            model = self.eps_model

        device = self.beta.device
        img = torch.randn(shape, device=device)

        for t in tqdm(
            reversed(range(0, self.n_steps)),
            desc="sampling loop time step",
            total=self.n_steps,
        ):
            t_tensor = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
            eps_theta = model(torch.cat([img, z], dim=-1), t_tensor)

            alpha_bar = self.gather(self.alpha_bar, t_tensor)
            alpha = self.gather(self.alpha, t_tensor)
            eps_coef = (1 - alpha) / (1 - alpha_bar).sqrt()
            mean = (img - eps_coef * eps_theta) / alpha.sqrt()
            var = self.gather(self.sigma2, t_tensor)

            if t > 0:
                img = mean + var.sqrt() * torch.randn_like(img)
            else:
                img = mean

        return img, mu, logvar

    def p_sample(self, xt, conditioning_vars, t):
        """
        Sample from p(x_{t-1} | x_t).

        Args:
            xt (torch.Tensor): Current data.
            conditioning_vars (dict): Conditioning variables.
            t (int): Current timestep.

        Returns:
            torch.Tensor: Sampled x_{t-1}.
        """
        t_tensor = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        c, mu, logvar = self.conditioning_module(conditioning_vars).to(self.device)
        return self.p_sample(xt, c, t_tensor)
