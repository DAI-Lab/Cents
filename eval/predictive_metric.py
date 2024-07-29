import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, t.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        y_hat = self.sigmoid(self.fc(output))
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - predictive_score: MAE of the predictions on the original data
    """
    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Build a post-hoc RNN predictive network
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    model = Predictor(input_dim=dim - 1, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    # Training using Synthetic dataset
    for itt in tqdm(range(iterations), desc="training", total=iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb = [generated_data[i][:-1, : (dim - 1)] for i in train_idx]
        T_mb = [generated_time[i] - 1 for i in train_idx]
        Y_mb = [generated_data[i][1:, (dim - 1)].reshape(-1, 1) for i in train_idx]

        X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
        T_mb = torch.tensor(np.array(T_mb), dtype=torch.int64).to(device)
        Y_mb = torch.tensor(np.array(Y_mb), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        y_pred = model(X_mb, T_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    # Test the trained model on the original data
    X_mb = [ori_data[i][:-1, : (dim - 1)] for i in range(no)]
    T_mb = [ori_time[i] - 1 for i in range(no)]
    Y_mb = [ori_data[i][1:, (dim - 1)].reshape(-1, 1) for i in range(no)]

    X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
    T_mb = torch.tensor(np.array(T_mb), dtype=torch.int64).to(device)
    Y_mb = torch.tensor(np.array(Y_mb), dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_mb, T_mb)

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].cpu().numpy(), y_pred[i].cpu().numpy())

    predictive_score = MAE_temp / no

    return predictive_score


def extract_time(data):
    """
    Extract time information from the input data.

    Args:
    - data: numpy array of shape (n_timeseries, n_timestamps, n_dimensions)

    Returns:
    - time: list of actual sequence lengths for each time series
    - max_seq_len: maximum sequence length across all time series
    """
    # Assume that zero padding is used for shorter sequences
    # Non-zero values in any dimension indicate a valid timestamp
    time = (data.sum(axis=2) != 0).sum(axis=1)
    max_seq_len = time.max()

    return time.tolist(), max_seq_len
