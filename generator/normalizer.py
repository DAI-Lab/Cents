import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from generator.conditioning import ConditioningModule

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StatsHead(nn.Module):
    """
    A small MLP that maps from the embedding z (from ConditioningModule)
    to [pred_mu[dims], pred_log_sigma[dims]] for n_dims dimensions.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, n_dims: int):
        super().__init__()
        self.n_dims = n_dims
        # Final layer must produce 2*n_dims
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_dims),  # 2 per dimension => mu + log_sigma
        )

    def forward(self, z: torch.Tensor):
        """
        z: (batch_size, embedding_dim)

        Returns:
          pred_mu: shape (batch_size, n_dims)
          pred_sigma: shape (batch_size, n_dims)
        """
        out = self.net(z)  # shape (batch_size, 2*n_dims)
        batch_size = out.size(0)

        # Split into mu and log_sigma
        out = out.view(batch_size, 2, self.n_dims)  # (batch_size, 2, n_dims)
        pred_mu = out[:, 0, :]  # (batch_size, n_dims)
        pred_log_sigma = out[:, 1, :]  # (batch_size, n_dims)
        pred_sigma = torch.exp(pred_log_sigma)
        return pred_mu, pred_sigma


class NormalizerModule(nn.Module):
    """
    ConditioningModule + StatsHead for multi-dimensional means/stds.
    """

    def __init__(self, cond_module: nn.Module, hidden_dim: int, n_dims: int):
        super().__init__()
        self.cond_module = cond_module
        self.embedding_dim = cond_module.embedding_dim
        self.n_dims = n_dims

        self.stats_head = StatsHead(
            embedding_dim=self.embedding_dim, hidden_dim=hidden_dim, n_dims=n_dims
        )

    def forward(self, cat_vars_dict: Dict[str, torch.Tensor]):
        """
        cat_vars_dict: {var_name -> LongTensor(batch_size,)}
        Returns: pred_mu, pred_sigma each (batch_size, n_dims)
        """
        z, _, _ = self.cond_module(cat_vars_dict, sample=False)
        pred_mu, pred_sigma = self.stats_head(z)
        return pred_mu, pred_sigma


class Normalizer:
    """
    A class that:
      1) Holds a reference to a dataset that has integer-coded context variables
         + multiple time series columns (e.g., dataset.time_series_column_names).
      2) Computes row-level dimension-wise (mean,std), then aggregates by context -> group_stats.
      3) Trains a NormalizerModule (ConditioningModule + multi-dim StatsHead) to predict (mu_array, sigma_array).
      4) Transforms or inverse-transforms data row-by-row, dimension-wise, using the learned model.
      5) Can save/load the model checkpoint.

    Retains config handling and method signatures from the previous single-dim logic.
    """

    def __init__(self, dataset, cfg):
        """
        Args:
            dataset: A TimeSeriesDataset or similar, with .data (DataFrame)
                     plus a list of dimension columns in .time_series_column_names.
            cfg: A config object with fields like device, run_dir, etc.
        """
        self.dataset_cfg = cfg
        self.normalizer_cfg = self._init_normalizer_config()
        self.dataset = dataset
        self.context_vars = dataset.conditioning_vars  # e.g. ["month", "weekday", ...]
        self.time_series_cols = dataset.time_series_column_names  # multi-dim columns
        self.n_dims = len(self.time_series_cols)

        self.device = self.normalizer_cfg.device
        self.group_stats = {}

        # Build the conditioning module
        cond_module = ConditioningModule(
            self.dataset_cfg.conditioning_vars,
            self.normalizer_cfg.embedding_dim,
            self.normalizer_cfg.device,
        )

        # Build the combined NormalizerModule for n_dims
        self.normalizer_model = NormalizerModule(
            cond_module=cond_module,
            hidden_dim=self.normalizer_cfg.hidden_dim,
            n_dims=self.n_dims,
        ).to(self.normalizer_cfg.device)

        self.optim = None

    def compute_group_stats(self):
        """
        For each row, compute dimension-wise row_mean, row_std across
        self.time_series_cols, then group by context -> average dimension-wise.
        """
        df = self.dataset.data.copy()

        # Per-row dimension arrays
        row_means_list = []
        row_stds_list = []

        for i, row in df.iterrows():
            dims_mean = []
            dims_std = []
            # Loop over each dimension column
            for col_name in self.time_series_cols:
                arr = np.array(row[col_name], dtype=np.float32).flatten()
                dims_mean.append(arr.mean())
                dims_std.append(arr.std())
            row_means_list.append(np.array(dims_mean, dtype=np.float32))
            row_stds_list.append(np.array(dims_std, dtype=np.float32))

        df["row_means_array"] = row_means_list
        df["row_stds_array"] = row_stds_list

        # Group by context vars, average dimension arrays
        grouped_stats = {}
        for group_vals, group_df in df.groupby(self.context_vars):
            # shape (n_samples_in_group,) each an array of shape (n_dims,)
            mean_arrays = list(group_df["row_means_array"])
            std_arrays = list(group_df["row_stds_array"])

            mean_arrays = np.stack(mean_arrays, axis=0)  # (n_group, n_dims)
            std_arrays = np.stack(std_arrays, axis=0)  # (n_group, n_dims)

            mu_array = mean_arrays.mean(axis=0)  # shape (n_dims,)
            sigma_array = std_arrays.mean(axis=0)  # shape (n_dims,)

            grouped_stats[tuple(group_vals)] = (mu_array, sigma_array)

        self.group_stats = grouped_stats

    def create_training_dataset(self):
        """
        Yields (cat_vars_dict, mu_array, sigma_array) with shape (n_dims,).
        """
        if not self.group_stats:
            raise ValueError(
                "No group_stats found. Please run compute_group_stats() first."
            )

        data_tuples = []
        for ctx_tuple, (mu_arr, sigma_arr) in self.group_stats.items():
            data_tuples.append((ctx_tuple, mu_arr, sigma_arr))

        class _TrainSet(Dataset):
            def __init__(self, data_tuples, context_vars, n_dims):
                super().__init__()
                self.samples = data_tuples
                self.context_vars = context_vars
                self.n_dims = n_dims

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                ctx_tuple, mu_array, sigma_array = self.samples[idx]
                cat_vars_dict = {}
                for i, var_name in enumerate(self.context_vars):
                    cat_vars_dict[var_name] = torch.tensor(
                        ctx_tuple[i], dtype=torch.long
                    )
                # Convert dimension arrays to Tensors => shape (n_dims,)
                mu_t = torch.from_numpy(mu_array).float()
                sigma_t = torch.from_numpy(sigma_array).float()
                return cat_vars_dict, mu_t, sigma_t

        return _TrainSet(data_tuples, self.context_vars, self.n_dims)

    def train_normalizer(self):
        """
        Train the normalizer model to predict (mu_array, sigma_array) for n_dims dims,
        assuming the DataLoader already produces:
        cat_vars_dict: {var_name -> LongTensor(batch_size,)}
        mu_tensor: shape (batch_size, n_dims)
        sigma_tensor: shape (batch_size, n_dims)
        """
        train_dataset = self.create_training_dataset()
        # The DataLoader must be returning (cat_vars_dict, mu_tensor, sigma_tensor)
        # with the shapes described above.
        train_loader = DataLoader(
            train_dataset, batch_size=self.normalizer_cfg.batch_size, shuffle=True
        )

        self.optim = torch.optim.Adam(
            self.normalizer_model.parameters(), lr=self.normalizer_cfg.lr
        )

        self.normalizer_model.train()
        for epoch in range(self.normalizer_cfg.n_epochs):
            epoch_loss = 0.0

            for cat_vars_dict, mu_tensor, sigma_tensor in train_loader:
                for var_name in cat_vars_dict:
                    cat_vars_dict[var_name] = cat_vars_dict[var_name].to(self.device)

                mu_tensor = mu_tensor.to(self.device)  # (batch_size, n_dims)
                sigma_tensor = sigma_tensor.to(self.device)  # (batch_size, n_dims)

                self.optim.zero_grad()
                pred_mu, pred_sigma = self.normalizer_model(cat_vars_dict)

                # 3) Compute losses
                loss_mu = F.mse_loss(pred_mu, mu_tensor)
                loss_sigma = F.mse_loss(pred_sigma, sigma_tensor)
                loss = loss_mu + loss_sigma
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()
            print(f"epoch {epoch}/{self.normalizer_cfg.n_epochs}: Loss: {epoch_loss}")

    def transform(self, use_model: bool = False) -> pd.DataFrame:
        """
        For each row, dimension-wise:
          - Build cat_vars_dict
          - Get (mu_array, sigma_array) from model or group_stats
          - For each dimension col, do (col_data - mu[d]) / sigma[d].
        """
        df = self.dataset.data.copy()
        if use_model and (self.normalizer_model is None):
            raise ValueError(
                "No trained normalizer_model found. Did you call train_normalizer()?"
            )

        for i, row in df.iterrows():
            # Build cat_vars
            cat_vars_dict = {}
            for var_name in self.context_vars:
                cat_vars_dict[var_name] = torch.tensor(
                    row[var_name], dtype=torch.long, device=self.device
                ).unsqueeze(0)

            if use_model and self.normalizer_model is not None:
                with torch.no_grad():
                    pred_mu, pred_sigma = self.normalizer_model(cat_vars_dict)
                mu_arr = pred_mu[0].cpu().numpy()  # shape (n_dims,)
                sigma_arr = pred_sigma[0].cpu().numpy()
            else:
                ctx_tuple = tuple(row[var] for var in self.context_vars)
                if ctx_tuple in self.group_stats:
                    mu_arr, sigma_arr = self.group_stats[ctx_tuple]
                else:
                    mu_arr = np.zeros(self.n_dims, dtype=np.float32)
                    sigma_arr = np.ones(self.n_dims, dtype=np.float32)

            # Dimension-wise transform
            for dim_idx, col_name in enumerate(self.time_series_cols):
                arr = np.array(row[col_name], dtype=np.float32)
                arr_norm = (arr - mu_arr[dim_idx]) / (sigma_arr[dim_idx] + 1e-8)
                df.at[i, col_name] = arr_norm

        return df

    def inverse_transform(
        self, df: pd.DataFrame, use_model: bool = True
    ) -> pd.DataFrame:
        """
        Inverse dimension-wise transform:
          arr_orig = arr_norm * sigma[d] + mu[d].
        """
        if use_model and (self.normalizer_model is None):
            raise ValueError(
                "No trained normalizer_model found. Did you call train_normalizer()?"
            )

        for i, row in df.iterrows():
            cat_vars_dict = {}
            for var_name in self.context_vars:
                cat_vars_dict[var_name] = torch.tensor(
                    row[var_name], dtype=torch.long, device=self.device
                ).unsqueeze(0)

            if use_model and self.normalizer_model is not None:
                with torch.no_grad():
                    pred_mu, pred_sigma = self.normalizer_model(cat_vars_dict)
                mu_arr = pred_mu[0].cpu().numpy()
                sigma_arr = pred_sigma[0].cpu().numpy()
            else:
                ctx_tuple = tuple(row[var] for var in self.context_vars)
                if ctx_tuple in self.group_stats:
                    mu_arr, sigma_arr = self.group_stats[ctx_tuple]
                else:
                    mu_arr = np.zeros(self.n_dims, dtype=np.float32)
                    sigma_arr = np.ones(self.n_dims, dtype=np.float32)

            for dim_idx, col_name in enumerate(self.time_series_cols):
                arr_norm = np.array(row[col_name], dtype=np.float32)
                arr_orig = arr_norm * (sigma_arr[dim_idx] + 1e-8) + mu_arr[dim_idx]
                df.at[i, col_name] = arr_orig

        return df

    def save(self, path: str = None, epoch: int = None):
        if path is None:
            hydra_output_dir = os.path.join(self.cfg.run_dir)
            os.makedirs(os.path.join(hydra_output_dir, "checkpoints"), exist_ok=True)
            path = os.path.join(
                hydra_output_dir,
                "checkpoints",
                f"normalizer_checkpoint_{epoch if epoch else 'final'}.pt",
            )

        checkpoint = {
            "epoch": epoch if epoch is not None else 0,
            "normalizer_model_state": self.normalizer_model.state_dict(),
        }

        torch.save(checkpoint, path)
        print(f"Saved Normalizer checkpoint to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.normalizer_model.load_state_dict(checkpoint["normalizer_model_state"])
        print(f"Loaded Normalizer from {path}")

    def _init_normalizer_config(self):
        config_path = os.path.join(ROOT_DIR, "config", "normalizer", "default.yaml")
        normalizer_cfg = OmegaConf.load(config_path)
        return normalizer_cfg
