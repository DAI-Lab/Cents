"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)  # Predict all dimensions

    def forward(self, x, t):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, t.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat = self.fc(output)
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    no, seq_len, dim = ori_data.shape

    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = max(int(dim / 2), 1)
    iterations = 5000
    batch_size = 128

    model = Predictor(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    for itt in tqdm(
        range(iterations), desc="Training Predictive Score Model", total=iterations
    ):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb = [
            generated_data[i][:-1, :] for i in train_idx
        ]  # Use all dimensions for input
        T_mb = [generated_time[i] - 1 for i in train_idx]
        Y_mb = [
            generated_data[i][1:, :].reshape(-1, dim) for i in train_idx
        ]  # Predict all dimensions

        X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
        T_mb = torch.tensor(np.array(T_mb), dtype=torch.int64).to(device)
        Y_mb = torch.tensor(np.array(Y_mb), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        y_pred = model(X_mb, T_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    X_mb = [ori_data[i][:-1, :] for i in range(no)]
    T_mb = [ori_time[i] - 1 for i in range(no)]
    Y_mb = [ori_data[i][1:, :].reshape(-1, dim) for i in range(no)]

    X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
    T_mb = torch.tensor(np.array(T_mb), dtype=torch.int64).to(device)
    Y_mb = torch.tensor(np.array(Y_mb), dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_mb, T_mb)

    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), y_pred[i].cpu().numpy())

    predictive_score = MAE_temp / no

    return predictive_score


def extract_time(data):
    time = (data.sum(axis=2) != 0).sum(axis=1)
    max_seq_len = time.max()
    return time.tolist(), max_seq_len
