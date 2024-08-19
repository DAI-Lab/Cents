import numpy as np
import scipy.signal as sig
import torch.utils.data
from torch import nn
from tqdm import tqdm

from data_utils.dataset import prepare_dataloader
from generator.diffcharge.network import *
from generator.diffusion_ts.gaussian_diffusion import cosine_beta_schedule


class DDPM:
    def __init__(self, opt):
        super().__init__()
        if opt.network == "attention":
            self.eps_model = Attention(opt).to(opt.device)
        else:
            self.eps_model = CNN(opt).to(opt.device)
        self.opt = opt
        self.n_steps = opt.n_steps
        if opt.schedule == "linear":
            self.beta = torch.linspace(
                opt.beta_start, opt.beta_end, opt.n_steps, device=opt.device
            )
        elif opt.schedule == "cosine":
            self.beta = cosine_beta_schedule(opt.n_steps)
        else:
            self.beta = (
                torch.linspace(
                    opt.beta_start**0.5,
                    opt.beta_end**0.5,
                    opt.n_steps,
                    device=opt.device,
                )
                ** 2
            )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # self.sigma2 = self.beta
        self.sigma2 = torch.cat(
            (
                torch.tensor([self.beta[0]], device=opt.device),
                self.beta[1:] * (1 - self.alpha_bar[0:-1]) / (1 - self.alpha_bar[1:]),
            )
        )
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=opt.init_lr)
        self.loss_func = nn.MSELoss()
        p1, p2 = int(0.75 * opt.n_epochs), int(0.9 * opt.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar**0.5) * x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5) * eps

    def p_sample(self, xt, c, t):
        eps_theta = self.eps_model(xt, c, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = (xt - eps_coef * eps_theta) / (alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5) * z

    def cal_loss(self, x0, c):  # (B, L, 1)
        batch_size = x0.shape[0]
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, c, t)
        return self.loss_func(noise, eps_theta)

    def sample(self, weight_path, n_samples, condition, smooth=True):
        c = condition.to(self.opt.device)

        with torch.no_grad():
            self.eps_model.eval()
            x = torch.randn([n_samples, self.opt.seq_len, self.opt.input_dim]).to(
                self.opt.device
            )
            for j in tqdm(
                range(0, self.n_steps, 1), desc=f"Sampling steps of {self.n_steps}"
            ):
                t = torch.ones(n_samples, dtype=torch.long).to(self.opt.device) * (
                    self.n_steps - j - 1
                )
                x = self.p_sample(x, c, t)

            if smooth:
                for i in range(n_samples):
                    filtered_x = sig.medfilt(x[i].cpu().numpy(), kernel_size=(5, 1))
                    x[i] = torch.tensor(filtered_x, dtype=torch.float32).to(
                        self.opt.device
                    )

            return x

    def train(self, dataset):
        # self.train()
        batch_size = self.opt.batch_size
        epoch_loss = []
        train_loader = prepare_dataloader(dataset, batch_size)

        for epoch in range(self.opt.n_epochs):
            batch_loss = []
            for i, (time_series_batch, month_label_batch, day_label_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            ):
                x0 = time_series_batch
                c = torch.cat(
                    [day_label_batch.unsqueeze(1), month_label_batch.unsqueeze(1)],
                    dim=1,
                ).to(self.opt.device)
                self.optimizer.zero_grad()
                loss = self.cal_loss(x0, c)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
            print(f"epoch={epoch}/{self.opt.n_epochs}, loss={epoch_loss[-1]}")
            self.lr_scheduler.step()

    def generate(self, day_labels, month_labels):
        num_samples = day_labels.shape[0]
        shape = (num_samples, self.opt.seq_len, self.opt.input_dim)
        return self._generate(shape, [day_labels, month_labels])

    def _generate(self, shape, labels):
        with torch.no_grad():
            c = torch.cat([label.unsqueeze(1) for label in labels], dim=1).to(
                self.opt.device
            )
            samples = self.sample(None, n_samples=shape[0], condition=c, smooth=True)
            return samples
