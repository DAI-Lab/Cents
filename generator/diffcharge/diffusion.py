"""
This class is adapted/taken from the Diffusion_TS GitHub repository:

Repository: https://github.com/Y-debug-sys/Diffusion-TS
Author: Xinyu Yuan
License: MIT License

Modifications:
- Conditioning and sampling logic
- Added further functions and removed unused functionality
- Added conditioning module logic for rare and non-rare samples
- Implemented saving and loading functionality

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import copy
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from datasets.utils import prepare_dataloader
from generator.conditioning import ConditioningModule
from generator.diffcharge.network import CNN
from generator.diffcharge.network import Attention
from generator.diffusion_ts.gaussian_diffusion import cosine_beta_schedule
from generator.diffusion_ts.model_utils import default
from generator.diffusion_ts.model_utils import extract
from generator.diffusion_ts.model_utils import identity


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
    Exponential Moving Average (EMA) of model parameters.
    """

    def __init__(self, model, beta, update_every, device):
        """
        Initialize the EMA class.

        Args:
            model (nn.Module): The model to apply EMA to.
            beta (float): The decay rate for EMA.
            update_every (int): Update EMA every 'update_every' steps.
            device (torch.device): Device to store the EMA model.
        """
        self.model = model
        self.ema_model = copy.deepcopy(model).eval().to(device)
        self.beta = beta
        self.update_every = update_every
        self.step = 0
        self.device = device
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self):
        """
        Update the EMA parameters.
        """
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
        """
        Forward pass using EMA model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output from EMA model.
        """
        return self.ema_model(x)


class DDPM(nn.Module):
    def __init__(self, opt):
        super(DDPM, self).__init__()
        self.opt = opt
        self.device = opt.device

        # Initialize the conditioning module
        self.conditioning_module = ConditioningModule(
            categorical_dims=opt.categorical_dims,
            embedding_dim=opt.cond_emb_dim,
            device=opt.device,
        ).to(self.device)

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
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.loss_func = nn.MSELoss()
        n_epochs = opt.n_epochs
        p1, p2 = int(0.75 * n_epochs), int(0.9 * n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

        # Initialize EMA
        self.ema = EMA(
            self.eps_model,
            beta=opt.ema_decay,
            update_every=opt.ema_update_interval,
            device=self.device,
        )

        # Initialize tracking variables
        self.current_epoch = 0
        self.writer = SummaryWriter(log_dir=os.path.join("runs", "ddpm"))

    def gather(self, const, t):
        """
        Gather specific timestep constants.

        Args:
            const (torch.Tensor): Constants tensor.
            t (torch.Tensor): Timestep tensor.

        Returns:
            torch.Tensor: Gathered constants.
        """
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        """
        Compute mean and variance for q(x_t | x_0).

        Args:
            x0 (torch.Tensor): Original data.
            t (torch.Tensor): Timesteps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance.
        """
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar.sqrt()) * x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        """
        Sample from q(x_t | x_0).

        Args:
            x0 (torch.Tensor): Original data.
            t (torch.Tensor): Timesteps.
            eps (torch.Tensor): Noise.

        Returns:
            torch.Tensor: Sampled x_t.
        """
        mean, var = self.q_xt_x0(x0, t)
        return mean + var.sqrt() * eps

    def p_sample(self, xt, c, t, guidance_scale=1.0):
        """
        Sample from p(x_{t-1} | x_t).

        Args:
            xt (torch.Tensor): Current data.
            c (torch.Tensor): Conditioning variables.
            t (torch.Tensor): Current timestep.
            guidance_scale (float): Guidance scale for conditional generation.

        Returns:
            torch.Tensor: Sampled x_{t-1}.
        """
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
        """
        Calculate the training loss.

        Args:
            x0 (torch.Tensor): Original data.
            c (torch.Tensor): Conditioning variables.
            drop_prob (float): Probability to drop conditioning for augmentation.

        Returns:
            torch.Tensor: Computed loss.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)

        # Randomly drop conditioning for augmentation
        if torch.rand(1).item() < drop_prob:
            c = torch.zeros_like(c)

        eps_theta = self.eps_model(xt, c, t)
        return self.loss_func(noise, eps_theta)

    def train_model(self, train_dataset):
        """
        Train the DDPM model.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
        """
        self.train()
        self.to(self.device)

        train_loader = prepare_dataloader(
            train_dataset, self.opt.batch_size, shuffle=True
        )

        os.makedirs(self.opt.results_folder, exist_ok=True)

        for epoch in tqdm(range(self.opt.n_epochs), desc="Training"):
            self.train_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_epoch = epoch + 1
            batch_loss = []
            for i, (time_series_batch, conditioning_vars_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                x0 = time_series_batch.to(self.device)
                c = self.conditioning_module(conditioning_vars_batch).to(self.device)

                # Compute rare_mask after warm-up epochs
                if epoch >= self.opt.warm_up_epochs:
                    with torch.no_grad():
                        if self.opt.freeze_cond_after_warmup:
                            for param in self.conditioning_module.parameters():
                                param.requires_grad = (
                                    False  # Freeze conditioning module
                                )

                        batch_embeddings = self.conditioning_module(
                            conditioning_vars_batch
                        )
                        self.conditioning_module.update_running_statistics(
                            batch_embeddings
                        )
                        rare_mask = (
                            self.conditioning_module.is_rare(batch_embeddings)
                            .to(self.device)
                            .float()
                        )
                else:
                    rare_mask = torch.zeros((x0.size(0),), device=self.device)

                self.optimizer.zero_grad()

                if epoch < self.opt.warm_up_epochs:
                    # Standard loss without separating rare and non-rare
                    loss = self.cal_loss(x0, c, drop_prob=0.1)
                else:
                    # Separate loss for rare and non-rare samples
                    rare_indices = (rare_mask == 1.0).nonzero(as_tuple=True)[0]
                    non_rare_indices = (rare_mask == 0.0).nonzero(as_tuple=True)[0]

                    loss_rare = torch.tensor(0.0, device=self.device)
                    loss_non_rare = torch.tensor(0.0, device=self.device)

                    if len(rare_indices) > 0:
                        x0_rare = x0[rare_indices]
                        c_rare = c[rare_indices]
                        loss_rare = self.cal_loss(x0_rare, c_rare, drop_prob=0.0)

                    if len(non_rare_indices) > 0:
                        x0_non_rare = x0[non_rare_indices]
                        c_non_rare = c[non_rare_indices]
                        loss_non_rare = self.cal_loss(
                            x0_non_rare, c_non_rare, drop_prob=0.0
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

                # Optional: Logging per batch
                # self.writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

            epoch_mean_loss = sum(batch_loss) / len(batch_loss)
            print(f"Epoch {epoch + 1}/{self.opt.n_epochs}, Loss: {epoch_mean_loss:.4f}")

            # Scheduler step
            self.lr_scheduler.step(epoch_mean_loss)

            if (epoch + 1) % self.opt.save_cycle == 0:
                os.mkdir(os.path.join(self.opt.results_folder, self.train_timestamp))

                checkpoint_path = os.path.join(
                    os.path.join(self.opt.results_folder, self.train_timestamp),
                    f"diffcharge_checkpoint_{epoch + 1}.pt",
                )

                self.save(checkpoint_path, self.current_epoch)

        print("Training complete")
        self.writer.close()

    def save(self, path: str, epoch: int = None):
        """
        Save the DDPM model, optimizer, EMA state, and epoch number.

        Args:
            path (str): The file path to save the checkpoint to.
            epoch (int, optional): The current epoch number. Defaults to None.
        """
        checkpoint = {
            "epoch": epoch if epoch is not None else self.current_epoch,
            "eps_model_state_dict": self.eps_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_state_dict": self.ema.ema_model.state_dict(),
            "alpha_bar": self.alpha_bar.cpu(),
            "beta": self.beta.cpu(),
        }
        torch.save(checkpoint, path)
        print(f"Saved DDPM checkpoint to {path}")

    def load(self, path: str):
        """
        Load the DDPM model, optimizer, EMA state, and epoch number from a checkpoint file.

        Args:
            path (str): The file path to load the checkpoint from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Load epsilon model state
        if "eps_model_state_dict" in checkpoint:
            self.eps_model.load_state_dict(checkpoint["eps_model_state_dict"])
            print("Loaded epsilon model state.")
        else:
            raise KeyError("Checkpoint does not contain 'eps_model_state_dict'.")

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print("No optimizer state found in checkpoint.")

        # Load EMA state
        if "ema_state_dict" in checkpoint:
            self.ema.ema_model.load_state_dict(checkpoint["ema_state_dict"])
            print("Loaded EMA model state.")
        else:
            print("No EMA state found in checkpoint.")

        # Load alpha_bar and beta if needed
        if "alpha_bar" in checkpoint:
            self.alpha_bar = checkpoint["alpha_bar"].to(self.device)
            print("Loaded alpha_bar.")
        if "beta" in checkpoint:
            self.beta = checkpoint["beta"].to(self.device)
            print("Loaded beta.")

        # Load epoch number
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
            print(f"Loaded epoch number: {self.current_epoch}")
        else:
            print("No epoch information found in checkpoint.")

        # Move models to the correct device
        self.eps_model.to(self.device)
        self.conditioning_module.to(self.device)
        self.ema.ema_model.to(self.device)
        print(f"DDPM models moved to {self.device}.")

    def sample_conditioning_vars(self, dataset, batch_size, random=False):
        """
        Sample conditioning variables from the dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            batch_size (int): Number of samples to generate.
            random (bool): Whether to sample randomly or from the dataset.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of conditioning variables.
        """
        conditioning_vars = {}
        if random:
            for var_name, num_categories in self.opt.categorical_dims.items():
                conditioning_vars[var_name] = torch.randint(
                    0,
                    num_categories,
                    (batch_size,),
                    dtype=torch.long,
                    device=self.device,
                )
        else:
            sampled_rows = dataset.data.sample(n=batch_size).reset_index(drop=True)
            for var_name in self.opt.categorical_dims.keys():
                conditioning_vars[var_name] = torch.tensor(
                    sampled_rows[var_name].values, dtype=torch.long, device=self.device
                )

        return conditioning_vars

    def generate(self, conditioning_vars: dict, use_ema_sampling: bool = False):
        """
        Generate synthetic time series data using the trained model.

        Args:
            conditioning_vars (dict): Conditioning variables for generation.
            use_ema_sampling (bool, optional): Whether to use EMA model for generation. Defaults to False.

        Returns:
            torch.Tensor: Generated synthetic time series data.
        """
        num_samples = conditioning_vars[next(iter(conditioning_vars))].shape[0]
        shape = (num_samples, self.opt.seq_len, self.opt.input_dim)

        if use_ema_sampling:
            print("Generating using EMA model parameters.")
            with torch.no_grad():
                samples = self.ema.ema_model.sample(shape, conditioning_vars)
            return samples.cpu().numpy()
        else:
            print("Generating using regular epsilon model parameters.")
            return self._generate(shape, conditioning_vars).cpu().numpy()

    @torch.no_grad()
    def _generate(self, shape, conditioning_vars):
        """
        Internal method to generate samples using the standard sampling procedure.

        Args:
            shape (tuple): Shape of the samples to generate.
            conditioning_vars (dict): Conditioning variables for generation.

        Returns:
            torch.Tensor: Generated samples.
        """
        device = self.beta.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.n_steps)),
            desc="sampling loop time step",
            total=self.n_steps,
        ):
            img = self.p_sample(img, conditioning_vars, t)
        return img
