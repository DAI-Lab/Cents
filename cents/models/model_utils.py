"""
This class is adapted/taken from the Diffusion_TS GitHub repository:

Repository: https://github.com/Y-debug-sys/Diffusion-TS
Author: Xinyu Yuan
License: MIT License

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn


def _nan_check(t: Optional[torch.Tensor], name: str, extra: str = "") -> None:
    """Print location and stats when tensor contains NaN or Inf (for debugging)."""
    if t is None or not isinstance(t, torch.Tensor):
        return
    if not (torch.isnan(t).any() or torch.isinf(t).any()):
        return
    nan_c = torch.isnan(t).sum().item()
    inf_c = torch.isinf(t).sum().item()
    finite = t[~(torch.isnan(t) | torch.isinf(t))]
    min_s = finite.min().item() if finite.numel() > 0 else float("nan")
    max_s = finite.max().item() if finite.numel() > 0 else float("nan")
    mean_s = finite.float().mean().item() if finite.numel() > 0 else float("nan")
    print(
        f"[NaN/Inf] Transformer {name}: shape={tuple(t.shape)}, nan_count={nan_c}, inf_count={inf_c}, "
        f"finite_min={min_s:.6g}, finite_max={max_s:.6g}, finite_mean={mean_s:.6g} {extra}".strip()
    )


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """
    Create a linear schedule of betas for diffusion noise levels.
    Args:

        timesteps: Number of diffusion steps (T).

    Returns:
        Tensor of shape (timesteps,) with betas increasing linearly.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.004) -> torch.Tensor:
    """
    Create a cosine schedule of betas for diffusion noise levels.

    Args:
        timesteps: Number of diffusion steps (T).
        s: Small offset to avoid singularities.

    Returns:
        Tensor of shape (timesteps,) with betas computed via a cosine schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

import math
import torch


def cosine_beta_schedule_logsnr(
    timesteps: int,
    logsnr_min: float = -8.0,
    logsnr_max: float = 15.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float32)
    logsnr = logsnr_max + 0.5 * (logsnr_min - logsnr_max) * (1 - torch.cos(math.pi * t))

    alpha_bar = torch.sigmoid(logsnr).clamp(eps, 1.0 - eps)

    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


# normalization functions


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# learnable positional embeds


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(1, max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1
        )
        res = x - moving_mean
        return res, moving_mean


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1)
        if self.conv.weight.requires_grad:
            self.conv.weight.register_hook(lambda grad: grad.contiguous())
        self.drop = nn.Dropout(p=resid_pdrop)

    def forward(self, x):
        # x: (B, T, C)
        _nan_check(x, "Conv_MLP forward x (initial)")
        # # Print when values are extreme (even if not NaN) to debug downstream NaN
        # print(
        #     f"[Conv_MLP] x (initial): shape={tuple(x.shape)}, min={x.min().item():.6g}, max={x.max().item():.6g}, "
        #     f"abs_max={x.abs().max().item():.6g}, has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}"
        # )
        x = x.transpose(1, 2).contiguous()   # (B, C, T) contiguous
        _nan_check(x, "Conv_MLP forward x (transposed)")
        x = self.conv(x)
        _nan_check(x, "Conv_MLP forward x (conv)")
        x = self.drop(x)
        _nan_check(x, "Conv_MLP forward x (drop)")
        out = x.transpose(1, 2).contiguous()  # back to (B, T, C), contiguous
        _nan_check(out, "Conv_MLP forward out (transposed)")
        return out



class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(
                in_channels=n_embd,
                out_channels=int(mlp_hidden_times * n_embd),
                kernel_size=1,
                padding=0,
            ),
            act,
            nn.Conv1d(
                in_channels=int(mlp_hidden_times * n_embd),
                out_channels=int(mlp_hidden_times * n_embd),
                kernel_size=3,
                padding=1,
            ),
            act,
            nn.Conv1d(
                in_channels=int(mlp_hidden_times * n_embd),
                out_channels=n_embd,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)

        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear.weight)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, emb):
        # emb: (B, n_embd) - Pre-computed and combined (time + label)
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """

    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1
            ),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1),
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack(
            [lin_space ** float(p + 1) for p in range(trend_poly)], dim=0
        )

    def forward(self, input):
        b, c, h = input.shape
        if not input.is_contiguous():
            input = input.contiguous()
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """

    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """

    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        with torch.amp.autocast(device_type="cpu", enabled=False):
            x_freq = torch.fft.rfft(x.float(), dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq : -1]
            f = torch.fft.rfftfreq(t)[self.low_freq : -1]
        else:
            x_freq = x_freq[:, self.low_freq :]
            f = torch.fft.rfftfreq(t)[self.low_freq :]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2)).to(
            x_freq.device
        )
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float), "t -> () () t ()").to(
            x_freq.device
        )

        amp = rearrange(x_freq.abs(), "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, "b f t d -> b t d", "sum")

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(
            x_freq.abs(), top_k, dim=1, largest=True, sorted=True
        )
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij"
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """

    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(
            in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0
        )
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (
            (season_poly // 2, season_poly // 2)
            if season_poly % 2 == 0
            else (season_poly // 2, season_poly // 2 + 1)
        )
        s1 = torch.stack(
            [torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0
        )
        s2 = torch.stack(
            [torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0
        )
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        condition_embd,  # condition dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(encoder_output)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(encoder_output)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
    ):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = AdaLayerNorm(n_embd)
        self.attn = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        
        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, cond_emb, mask=None):
        # cond_emb is the combined time+label embedding
        a, att = self.attn(self.ln1(x, cond_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x, cond_emb))
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=4,
        block_activate="GELU",
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, input, cond_emb, padding_masks=None):
        x = input
        for block in self.blocks:
            x, _ = block(x, cond_emb, mask=padding_masks)
        return x


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
        condition_dim=1024,
    ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = AdaLayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        # Cross-attention for temporally-aligned dynamic context (same seq_len as target).
        # Conditioned via AdaLN on the same diffusion-timestep + static-context embedding
        # as the rest of the block, so global conditioning still flows through here.
        self.ln_dyn = AdaLayerNorm(n_embd)
        self.attn_dyn = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, cond_emb, dyn_ctx=None, mask=None):
        a, att = self.attn1(self.ln1(x, cond_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, cond_emb), encoder_output, mask=mask)
        x = x + a
        if dyn_ctx is not None:
            a, _ = self.attn_dyn(self.ln_dyn(x, cond_emb), dyn_ctx)
            x = x + a
        
        # FIX: chunk() returns views that are often non-contiguous. 
        # Since self.proj and self.trend use Conv1d, this causes the DDP stride mismatch.
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=1)
        
        # Make contiguous before passing to specialized blocks
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        
        trend, season = self.trend(x1), self.seasonal(x2)
        
        x = x + self.mlp(self.ln2(x, cond_emb))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        condition_dim=512,
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_feat=n_feat,
                    n_channel=n_channel,
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                    condition_dim=condition_dim,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, cond_emb, enc, dyn_ctx=None, padding_masks=None):
        b, c, _ = x.shape
        mean = []
        # Initialize accumulating tensors on the correct device
        season = torch.zeros((b, c, x.shape[-1]), device=x.device)
        # Note: FourierLayer returns same dim as input x (n_embd)

        trend = torch.zeros((b, c, self.blocks[0].linear.out_features), device=x.device)

        for block in self.blocks:
            x, residual_mean, residual_trend, residual_season = block(
                x, enc, cond_emb, dyn_ctx=dyn_ctx, mask=padding_masks
            )
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        max_len=2048,
        conv_params=None,
        cond_dim=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        
        self.cond_dim = cond_dim
        if cond_dim is not None:
            # Map static context embedding (B, cond_dim) -> (B, n_embd)
            self.cond_proj = nn.Linear(cond_dim, n_embd)
            # Map dynamic context sequence (B, T, cond_dim) -> (B, T, n_embd)
            self.dyn_ctx_proj = nn.Linear(cond_dim, n_embd)
        else:
            self.cond_proj = None
            self.dyn_ctx_proj = None

        self.time_emb = SinusoidalPosEmb(n_embd)

        self.cond_mix_mlp = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(
            n_embd,
            n_feat,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        self.combine_m = nn.Conv1d(
            n_layer_dec,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="circular",
            bias=False,
        )

        self.encoder = Encoder(
            n_layer_enc,
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
        )
        self.pos_enc = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len
        )

        self.decoder = Decoder(
            n_channel,
            n_feat,
            n_embd,
            n_heads,
            n_layer_dec,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
            condition_dim=n_embd,
        )
        self.pos_dec = LearnablePositionalEncoding(
            n_embd, dropout=resid_pdrop, max_len=max_len
        )

    def forward(self, input, t, padding_masks=None, return_res=False, cond=None, dyn_ctx=None):
        # cond: (B, cond_dim) static context or None
        # dyn_ctx: (B, T, cond_dim) dynamic context sequence or None
        # Ensure float32 so conv/linear (float32 params) never see double input
        if input.is_floating_point():
            input = input.float()
        if cond is not None and cond.is_floating_point():
            cond = cond.float()
        if dyn_ctx is not None and dyn_ctx.is_floating_point():
            dyn_ctx = dyn_ctx.float()
        _nan_check(input, "forward input")
        t_emb = self.time_emb(t)
        _nan_check(t_emb, "forward t_emb")

        label_emb = None
        if (cond is not None) and (self.cond_proj is not None):
            label_emb = self.cond_proj(cond)  # (B, n_embd)
            _nan_check(label_emb, "forward label_emb")
            total_cond_emb = self.cond_mix_mlp(torch.concat([t_emb, label_emb], dim=1))
        else:
            total_cond_emb = t_emb
        _nan_check(total_cond_emb, "forward total_cond_emb")

        # Project dynamic context sequence to n_embd for cross-attention
        dyn_ctx_emb = None
        if dyn_ctx is not None and self.dyn_ctx_proj is not None:
            dyn_ctx_emb = self.dyn_ctx_proj(dyn_ctx)  # (B, T, n_embd)
            _nan_check(dyn_ctx_emb, "forward dyn_ctx_emb")

        emb = self.emb(input)
        _nan_check(emb, "forward emb")
        inp_enc = self.pos_enc(emb)
        _nan_check(inp_enc, "forward inp_enc")

        enc_cond = self.encoder(inp_enc, total_cond_emb, padding_masks=padding_masks)
        _nan_check(enc_cond, "forward enc_cond")

        inp_dec = self.pos_dec(emb)
        _nan_check(inp_dec, "forward inp_dec")
        output, mean, trend, season = self.decoder(
            inp_dec, total_cond_emb, enc_cond, dyn_ctx=dyn_ctx_emb, padding_masks=padding_masks
        )
        _nan_check(output, "forward decoder output")
        _nan_check(mean, "forward decoder mean")
        _nan_check(trend, "forward decoder trend")
        _nan_check(season, "forward decoder season")

        res = self.inverse(output)
        _nan_check(res, "forward res (inverse output)")

        res_m = torch.mean(res, dim=1, keepdim=True).contiguous()
        _nan_check(res_m, "forward res_m")
        combine_m_out = self.combine_m(mean).contiguous()
        _nan_check(combine_m_out, "forward combine_m_out")
        combine_s_out = self.combine_s(season.transpose(1, 2)).transpose(1, 2).contiguous()
        _nan_check(combine_s_out, "forward combine_s_out")
        season_error = (combine_s_out + res - res_m).contiguous()
        _nan_check(season_error, "forward season_error")
        trend = (combine_m_out + res_m + trend).contiguous()
        _nan_check(trend, "forward trend (final)")

        if return_res:
            out_res = res - res_m
            _nan_check(out_res, "forward return res - res_m")
            return trend, combine_s_out, out_res

        return trend, season_error


def total_correlation(
    embeddings: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Mini-batch estimator of total correlation (β-TCVAE; Chen et al., 2018).

    Args:
        embeddings: Tensor of shape ``(batch, dim)`` containing latent codes.
        reduction: ``"mean"`` (default) divides by batch size; ``"sum"`` returns the raw sum.

    Returns:
        Scalar tensor with the estimated TC for the current mini-batch.
    """
    batch, _ = embeddings.shape
    log_qz_prob = torch.logsumexp(
        -0.5 * (embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).pow(2).sum(-1),
        dim=1,
    ) - math.log(batch)
    log_prod_qzi = (
        torch.logsumexp(
            -0.5 * (embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).pow(2), dim=1
        )
        - math.log(batch)
    ).sum(1)
    tc = (log_qz_prob - log_prod_qzi).sum()
    if reduction == "mean":
        tc = tc / batch
    return tc
