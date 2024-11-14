"""
This class is adapted/taken from the DiffCharge GitHub repository:

Repository: https://github.com/LSY-Cython/DiffCharge
Author: Siyang Li, Hui Xiong, Yize Chen (HKUST-GZ
License: None

Modifications (if any):
- Changes to conditioning logic
- Added classifier-free-guidance sampling

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import torch
from torch import nn


def time_embedding(t, hidden_dim, seq_len, device):  # (B, )
    t = t.view(-1, 1)
    te = torch.zeros(t.shape[0], hidden_dim).to(device)
    div_term = 1 / torch.pow(
        10000.0,
        torch.arange(0, hidden_dim, 2, dtype=torch.float32).to(device) / hidden_dim,
    )
    te[:, 0::2] = torch.sin(t * div_term)
    te[:, 1::2] = torch.cos(t * div_term)
    te = te.view(te.shape[0], 1, hidden_dim).repeat(1, seq_len, 1)  # (B, L, hidden_dim)
    return te


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
        self.cfg = cfg

        self.input_projector = nn.LSTM(
            cfg.input_dim + cfg.cond_emb_dim,
            cfg.model.hidden_dim,
            num_layers=4,
            batch_first=True,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model.hidden_dim,
            nhead=cfg.model.nhead,
            dim_feedforward=cfg.model.hidden_dim,
            activation="relu",
            batch_first=True,
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=5)
        self.output_projector = self.conv1d_with_init(
            cfg.model.hidden_dim, cfg.input_dim, 1
        )

    def conv1d_with_init(self, in_channels, out_channels, kernel_size):
        conv1d_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(conv1d_layer.weight)
        return conv1d_layer

    def forward(self, x, t):
        """
        Forward pass of the Attention model.

        Args:
            x (torch.Tensor): Input tensor with concatenated conditioning vectors. Shape: (B, L, input_dim + cond_emb_dim)
            t (torch.Tensor): Timesteps tensor. Shape: (B,)

        Returns:
            torch.Tensor: Output tensor. Shape: (B, L, input_dim)
        """
        hid_enc, _ = self.input_projector(x)  # (B, L, hidden_dim)

        time_emb = time_embedding(
            t, self.cfg.model.hidden_dim, self.cfg.seq_len, self.cfg.device
        )
        hid_enc = hid_enc + time_emb  # (B, L, hidden_dim)

        # Pass through Transformer Encoder
        trans_enc = self.trans_encoder(hid_enc)
        trans_enc = trans_enc.permute(0, 2, 1)
        output = self.output_projector(trans_enc).permute(0, 2, 1)

        return output


"""
This class is adapted/taken from the DiffCharge GitHub repository:

Repository: https://github.com/LSY-Cython/DiffCharge
Author: Siyang Li, Hui Xiong, Yize Chen (HKUST-GZ)
License: None

Modifications:
- Removed internal conditioning logic.
- Removed classifier-free-guidance sampling.
- Assumes conditioning vectors are concatenated externally to the input.
- Simplified forward method to accept only `x` and `t`.

Note: Please ensure compliance with the repository's license and credit the original authors when using or distributing this code.
"""

import torch
from torch import nn


def time_embedding(t, hidden_dim, seq_len, device):  # (B, )
    t = t.view(-1, 1)
    te = torch.zeros(t.shape[0], hidden_dim).to(device)
    div_term = 1 / torch.pow(
        10000.0,
        torch.arange(0, hidden_dim, 2, dtype=torch.float32).to(device) / hidden_dim,
    )
    te[:, 0::2] = torch.sin(t * div_term)
    te[:, 1::2] = torch.cos(t * div_term)
    te = te.view(te.shape[0], 1, hidden_dim).repeat(1, seq_len, 1)  # (B, L, hidden_dim)
    return te


class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.cfg = cfg
        self.input_projector = nn.LSTM(
            cfg.input_dim + cfg.cond_emb_dim,
            cfg.model.hidden_dim,
            num_layers=4,
            batch_first=True,
        )
        self.output_projector = nn.Sequential(
            nn.Conv1d(cfg.model.hidden_dim, cfg.model.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(cfg.model.hidden_dim),
            nn.Conv1d(cfg.model.hidden_dim, cfg.input_dim, kernel_size=1),
        )

    def forward(self, x, t):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor with concatenated conditioning vectors. Shape: (B, L, input_dim + cond_emb_dim)
            t (torch.Tensor): Timesteps tensor. Shape: (B,)

        Returns:
            torch.Tensor: Output tensor. Shape: (B, L, input_dim)
        """
        # Pass through LSTM
        hid_enc, _ = self.input_projector(x)  # (B, L, hidden_dim)

        # Add time embeddings
        time_emb = time_embedding(
            t, self.cfg.model.hidden_dim, self.cfg.seq_len, self.cfg.device
        )
        hid_enc = hid_enc + time_emb  # (B, L, hidden_dim)

        hid_enc = hid_enc.permute(0, 2, 1)
        output = self.output_projector(hid_enc).permute(0, 2, 1)
        return output
