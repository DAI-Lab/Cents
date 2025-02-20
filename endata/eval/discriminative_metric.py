"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_generator(
    data: np.ndarray, time: List[int], batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random batch of data and corresponding sequence lengths.

    Args:
        data (np.ndarray): The dataset, of shape (n_samples, seq_len, n_features).
        time (List[int]): List of sequence lengths for each sample.
        batch_size (int): Size of the batch to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Batch of data and corresponding sequence lengths.
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = np.array([data[i] for i in train_idx])
    T_mb = np.array([time[i] for i in train_idx])

    return X_mb, T_mb


class Discriminator(nn.Module):
    """
    A discriminator model for evaluating the similarity between original and generated time series data.

    Attributes:
        gru (nn.GRU): A GRU layer to process time series data.
        fc (nn.Linear): A fully connected layer to output classification logits.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initializes the Discriminator model.

        Args:
            input_dim (int): Dimensionality of the input time series data.
            hidden_dim (int): Number of hidden units in the GRU layer.
        """
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input time series data, shape (batch_size, seq_len, input_dim).
            t (torch.Tensor): Corresponding sequence lengths for the batch.

        Returns:
            torch.Tensor: Logits for the classification task.
        """
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))


def discriminative_score_metrics(
    ori_data: np.ndarray, generated_data: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes the discriminative score by training a discriminator to classify between original and generated data.

    Args:
        ori_data (np.ndarray): Original time series data, shape (n_samples, seq_len, n_features).
        generated_data (np.ndarray): Generated time series data, same shape as ori_data.

    Returns:
        Tuple[float, float, float]: Discriminative score, accuracy on generated data, and accuracy on original data.
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Extract time information
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max(ori_max_seq_len, generated_max_seq_len)

    # Network parameters
    hidden_dim = int(dim * 2)
    iterations = 2000
    batch_size = 128

    discriminator = Discriminator(dim, hidden_dim).to(device)
    optimizer = optim.Adam(discriminator.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # Train/test split
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

    # Training loop
    for _ in tqdm(
        range(iterations), desc="Training Discriminative Score Model", total=iterations
    ):
        # Batch generation
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb = torch.FloatTensor(X_mb).to(device)
        X_hat_mb = torch.FloatTensor(X_hat_mb).to(device)
        T_mb = torch.LongTensor(T_mb).to(device)
        T_hat_mb = torch.LongTensor(T_hat_mb).to(device)

        # Discriminator forward and backward pass
        optimizer.zero_grad()
        y_pred_real = discriminator(X_mb, T_mb)
        y_pred_fake = discriminator(X_hat_mb, T_hat_mb)

        loss_real = criterion(y_pred_real, torch.ones_like(y_pred_real))
        loss_fake = criterion(y_pred_fake, torch.zeros_like(y_pred_fake))
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

    # Testing the discriminator on the testing set
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

    # Compute accuracy and discriminative score
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    fake_acc = accuracy_score(np.zeros(len(y_pred_fake_curr)), (y_pred_fake_curr > 0.5))
    real_acc = accuracy_score(np.ones(len(y_pred_real_curr)), (y_pred_real_curr > 0.5))

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc


def extract_time(data: np.ndarray) -> Tuple[List[int], int]:
    """
    Extracts the sequence lengths for each sample in the dataset.

    Args:
        data (np.ndarray): Time series data, shape (n_samples, seq_len, n_features).

    Returns:
        Tuple[List[int], int]: List of sequence lengths and the maximum sequence length.
    """
    # Assume that zero padding is used for shorter sequences
    time = (data.sum(axis=2) != 0).sum(axis=1)
    max_seq_len = time.max()

    return time.tolist(), max_seq_len


def train_test_divide(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    ori_time: List[int],
    generated_time: List[int],
    test_ratio: float = 0.2,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
    List[int],
    List[int],
    List[int],
]:
    """
    Splits the original and generated data into training and testing sets.

    Args:
        ori_data (np.ndarray): Original data, shape (n_samples, seq_len, n_features).
        generated_data (np.ndarray): Generated data, same shape as ori_data.
        ori_time (List[int]): Sequence lengths for the original data.
        generated_time (List[int]): Sequence lengths for the generated data.
        test_ratio (float, optional): Proportion of data to use for testing. Defaults to 0.2.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int], List[int]]:
        Training and testing datasets for original and generated data, along with corresponding sequence lengths.
    """
    # Randomly shuffle the indices
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
