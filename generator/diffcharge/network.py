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
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.opt = opt
        self.cond_embedder = nn.Sequential(
            nn.Linear(opt.cond_dim * 2, opt.hidden_dim), nn.Tanh()
        )
        self.input_projector = nn.LSTM(
            opt.input_dim + opt.hidden_dim,
            opt.hidden_dim,
            num_layers=4,
            batch_first=True,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_dim, nhead=opt.nhead, dim_feedforward=opt.hidden_dim
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.output_projector = self.conv1d_with_init(opt.hidden_dim, opt.input_dim, 1)

        self.month_embed = nn.Embedding(12, opt.hidden_dim)
        self.weekday_embed = nn.Embedding(7, opt.hidden_dim)

    def conv1d_with_init(self, in_channels, out_channels, kernel_size):
        conv1d_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(conv1d_layer.weight)
        return conv1d_layer

    def forward(self, x, c, t):

        if self.opt.cond_flag == "conditional":
            weekday_emb = self.weekday_embed(c[:, 0])
            month_emb = self.month_embed(c[:, 1])
            cond_emb = torch.concat([weekday_emb, month_emb], dim=1)
            cond_emb = self.cond_embedder(cond_emb)  # (B, hidden_dim)
            cond_emb = cond_emb.view(cond_emb.shape[0], 1, self.opt.hidden_dim).repeat(
                1, self.opt.seq_len, 1
            )
            x = torch.cat([x, cond_emb], dim=2)

        hid_enc, (_, _) = self.input_projector(x)  # (B, L, hidden_dim)
        time_emb = time_embedding(
            t, self.opt.hidden_dim, self.opt.seq_len, self.opt.device
        )
        hid_enc = hid_enc + time_emb
        trans_enc = self.trans_encoder(hid_enc).permute(0, 2, 1)  # (B, hidden_dim, L)
        output = self.output_projector(trans_enc).permute(0, 2, 1)  # (B, L, input_dim)
        return output


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.opt = opt
        self.input_projector = nn.LSTM(
            opt.input_dim, opt.hidden_dim, num_layers=1, batch_first=True
        )
        self.output_projector = nn.Sequential(
            nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(opt.hidden_dim),
            nn.Conv1d(opt.hidden_dim, opt.input_dim, kernel_size=1),
        )

    def forward(self, x, c, t):
        hid_enc, (_, _) = self.input_projector(x)  # (B, L, hidden_dim)
        time_emb = time_embedding(
            t, self.opt.hidden_dim, self.opt.seq_len, self.opt.device
        )
        hid_enc = hid_enc + time_emb
        output = self.output_projector(hid_enc.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (B, L, input_dim)
        return output
