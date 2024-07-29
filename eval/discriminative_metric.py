import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


def discriminative_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim).to(device)
    optimizer = optim.Adam(discriminator.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # Train/test division for both original and generated data
    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training step
    for _ in tqdm(range(iterations), desc="training", total=iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb = torch.FloatTensor(X_mb).to(device)
        X_hat_mb = torch.FloatTensor(X_hat_mb).to(device)
        T_mb = torch.LongTensor(T_mb).to(device)
        T_hat_mb = torch.LongTensor(T_hat_mb).to(device)

        # Train discriminator
        optimizer.zero_grad()
        y_pred_real = discriminator(X_mb, T_mb)
        y_pred_fake = discriminator(X_hat_mb, T_hat_mb)

        loss_real = criterion(y_pred_real, torch.ones_like(y_pred_real))
        loss_fake = criterion(y_pred_fake, torch.zeros_like(y_pred_fake))
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

    # Test the performance on the testing set
    with torch.no_grad():
        test_x = torch.FloatTensor(test_x).to(device)
        test_x_hat = torch.FloatTensor(test_x_hat).to(device)
        test_t = torch.LongTensor(test_t).to(device)
        test_t_hat = torch.LongTensor(test_t_hat).to(device)

        y_pred_real_curr = torch.sigmoid(discriminator(test_x, test_t)).cpu().numpy()
        y_pred_fake_curr = (
            torch.sigmoid(discriminator(test_x_hat, test_t_hat)).cpu().numpy()
        )

    y_pred_final = np.squeeze(
        np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0)
    )
    y_label_final = np.concatenate(
        (np.ones(len(y_pred_real_curr)), np.zeros(len(y_pred_fake_curr)))
    )

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))

    fake_acc = accuracy_score(np.zeros(len(y_pred_fake_curr)), (y_pred_fake_curr > 0.5))
    real_acc = accuracy_score(np.ones(len(y_pred_real_curr)), (y_pred_real_curr > 0.5))

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc


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


def train_test_divide(
    ori_data, generated_data, ori_time, generated_time, test_ratio=0.2
):
    """
    Divide original and generated data into training and testing sets.

    Args:
    - ori_data: original data, shape (n_timeseries, n_timestamps, n_dimensions)
    - generated_data: synthetic data, same shape as ori_data
    - ori_time: list of sequence lengths for original data
    - generated_time: list of sequence lengths for generated data
    - test_ratio: proportion of data to use for testing (default: 0.2)

    Returns:
    - train_x, train_x_hat, test_x, test_x_hat: training and testing data for original and generated sets
    - train_t, train_t_hat, test_t, test_t_hat: corresponding time information for each set
    """
    # Determine split indices
    ori_idx = np.random.permutation(len(ori_data))
    generated_idx = np.random.permutation(len(generated_data))

    test_size = int(len(ori_data) * test_ratio)

    # Split original data
    train_x = ori_data[ori_idx[test_size:]]
    test_x = ori_data[ori_idx[:test_size]]
    train_t = [ori_time[i] for i in ori_idx[test_size:]]
    test_t = [ori_time[i] for i in ori_idx[:test_size]]

    # Split generated data
    train_x_hat = generated_data[generated_idx[test_size:]]
    test_x_hat = generated_data[generated_idx[:test_size]]
    train_t_hat = [generated_time[i] for i in generated_idx[test_size:]]
    test_t_hat = [generated_time[i] for i in generated_idx[:test_size]]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )
