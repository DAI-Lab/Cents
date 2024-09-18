import numpy as np
import scipy.signal as sig
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.utils import prepare_dataloader
from generator.conditioning import ConditioningModule
from generator.diffcharge.network import CNN
from generator.diffcharge.network import Attention
from generator.diffusion_ts.gaussian_diffusion import cosine_beta_schedule


class DDPM:
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = opt.device

        # Initialize the conditioning module
        self.conditioning_module = ConditioningModule(
            categorical_dims=opt.categorical_dims,
            embedding_dim=opt.cond_emb_dim,
            device=opt.device,
        )

        # Initialize the epsilon model
        if opt.network == "attention":
            self.eps_model = Attention(opt).to(self.device)
        else:
            self.eps_model = CNN(opt).to(self.device)

        self.n_steps = opt.n_steps
        schedule = opt.schedule
        beta_start = opt.beta_start
        beta_end = opt.beta_end

        if schedule == "linear":
            self.beta = torch.linspace(
                beta_start, beta_end, self.n_steps, device=self.device
            )
        elif schedule == "cosine":
            self.beta = cosine_beta_schedule(self.n_steps).to(self.device)
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
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.loss_func = nn.MSELoss()
        n_epochs = opt.n_epochs
        p1, p2 = int(0.75 * n_epochs), int(0.9 * n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

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

    def p_sample(self, xt, c, t, guidance_scale=1.0):
        eps_theta_cond = self.eps_model(xt, c, t)
        eps_theta_uncond = self.eps_model(xt, torch.zeros_like(c), t)
        eps_theta = eps_theta_uncond + guidance_scale * (
            eps_theta_cond - eps_theta_uncond
        )
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar).sqrt()
        mean = (xt - eps_coef * eps_theta) / alpha.sqrt()
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros_like(xt)
        else:
            z = torch.randn_like(xt)
        return mean + var.sqrt() * z

    def cal_loss(self, x0, c, drop_prob=0.15):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)

        # Randomly drop conditioning
        if torch.rand(1).item() < drop_prob:
            c = torch.zeros_like(c)

        eps_theta = self.eps_model(xt, c, t)
        return self.loss_func(noise, eps_theta)

    def train_model(self, dataset):
        batch_size = self.opt.batch_size
        epoch_loss = []
        train_loader = prepare_dataloader(dataset, batch_size)

        for epoch in range(self.opt.n_epochs):
            batch_loss = []
            for i, (time_series_batch, categorical_vars) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                x0 = time_series_batch.to(self.device)
                # Get conditioning vector
                c = self.conditioning_module(categorical_vars)
                self.optimizer.zero_grad()
                loss = self.cal_loss(x0, c, drop_prob=0.1)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
            print(f"epoch={epoch + 1}/{self.opt.n_epochs}, loss={epoch_loss[-1]}")
            self.lr_scheduler.step()

    def sample(self, n_samples, categorical_vars, smooth=True, guidance_scale=1.0):
        c = self.conditioning_module(categorical_vars).to(self.device)
        with torch.no_grad():
            self.eps_model.eval()
            x = torch.randn(n_samples, self.opt.seq_len, self.opt.input_dim).to(
                self.device
            )
            for j in tqdm(
                range(self.n_steps), desc=f"Sampling steps of {self.n_steps}"
            ):
                t = torch.full(
                    (n_samples,),
                    self.n_steps - j - 1,
                    dtype=torch.long,
                    device=self.device,
                )
                x = self.p_sample(x, c, t, guidance_scale=guidance_scale)
            if smooth:
                for i in range(n_samples):
                    filtered_x = sig.medfilt(x[i].cpu().numpy(), kernel_size=(5, 1))
                    x[i] = torch.tensor(filtered_x, dtype=torch.float32).to(self.device)
            return x

    def generate(self, categorical_vars):
        num_samples = categorical_vars[next(iter(categorical_vars))].shape[0]
        return self.sample(
            n_samples=num_samples,
            categorical_vars=categorical_vars,
            smooth=True,
            guidance_scale=self.opt.guidance_scale,
        )
