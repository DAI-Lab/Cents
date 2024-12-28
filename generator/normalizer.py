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
    to dimension-wise [mu, sigma] or [mu, sigma, z_min, z_max].
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, n_dims: int, do_scale: bool
    ):
        super().__init__()
        self.n_dims = n_dims
        self.do_scale = do_scale
        out_dim = 4 * n_dims if do_scale else 2 * n_dims

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor):
        """
        z: shape (batch_size, embedding_dim)

        Returns:
          pred_mu, pred_sigma (always shape (batch_size, n_dims))

          If self.do_scale == True, also returns pred_z_min, pred_z_max
          each (batch_size, n_dims). Otherwise, returns (None, None).
        """
        out = self.net(z)  # shape (batch_size, out_dim)
        batch_size = out.size(0)

        if self.do_scale:
            # 4 per dimension: [mu, log_sigma, z_min, z_max]
            out = out.view(batch_size, 4, self.n_dims)
            pred_mu = out[:, 0, :]  # (batch_size, n_dims)
            pred_log_sigma = out[:, 1, :]
            pred_z_min = out[:, 2, :]
            pred_z_max = out[:, 3, :]
        else:
            # 2 per dimension: [mu, log_sigma]
            out = out.view(batch_size, 2, self.n_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = None
            pred_z_max = None

        pred_sigma = torch.exp(pred_log_sigma)

        return pred_mu, pred_sigma, pred_z_min, pred_z_max


class NormalizerModule(nn.Module):
    """
    ConditioningModule + multi-dim StatsHead for
    [mu, sigma, (optionally) z_min, z_max].
    """

    def __init__(
        self, cond_module: nn.Module, hidden_dim: int, n_dims: int, do_scale: bool
    ):
        super().__init__()
        self.cond_module = cond_module
        self.embedding_dim = cond_module.embedding_dim
        self.stats_head = StatsHead(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            n_dims=n_dims,
            do_scale=do_scale,
        )

    def forward(self, cat_vars_dict: Dict[str, torch.Tensor]):
        """
        Returns:
          pred_mu, pred_sigma: always (batch_size, n_dims)
          pred_z_min, pred_z_max: (batch_size, n_dims) if do_scale=True else (None, None)
        """
        z, _, _ = self.cond_module(cat_vars_dict, sample=False)
        return self.stats_head(z)


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
        self.do_scale = cfg.scale
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
            do_scale=self.do_scale,
        ).to(self.normalizer_cfg.device)

        self.optim = None

    def compute_group_stats(self):
        """
        For each row, compute dimension-wise row_mean, row_std,
        then compute row_z = (arr - row_mean)/row_std for each dimension
        to find row_z_min, row_z_max if scale is True.
        Finally, group-average them by context.
        """
        df = self.dataset.data.copy()

        row_means_list = []
        row_stds_list = []
        row_z_min_list = []
        row_z_max_list = []

        for _, row in df.iterrows():
            dims_mean = []
            dims_std = []
            dims_z_min = []
            dims_z_max = []

            for col_name in self.time_series_cols:
                arr = np.array(row[col_name], dtype=np.float32).flatten()
                m = arr.mean()
                s = arr.std() + 1e-8
                dims_mean.append(m)
                dims_std.append(s)
                if self.do_scale:  # we only do z_min, z_max if scale=True
                    z = (arr - m) / s
                    dims_z_min.append(z.min())
                    dims_z_max.append(z.max())

            row_means_list.append(np.array(dims_mean, dtype=np.float32))
            row_stds_list.append(np.array(dims_std, dtype=np.float32))

            if self.do_scale:
                row_z_min_list.append(np.array(dims_z_min, dtype=np.float32))
                row_z_max_list.append(np.array(dims_z_max, dtype=np.float32))
            else:
                row_z_min_list.append(None)
                row_z_max_list.append(None)

        df["row_means_array"] = row_means_list
        df["row_stds_array"] = row_stds_list
        if self.do_scale:
            df["row_z_min_array"] = row_z_min_list
            df["row_z_max_array"] = row_z_max_list

        # Now group by context and average dimension arrays
        grouped_stats = {}
        for group_vals, group_df in df.groupby(self.context_vars):
            mean_arrays = np.stack(group_df["row_means_array"], axis=0)
            std_arrays = np.stack(group_df["row_stds_array"], axis=0)
            mu_array = mean_arrays.mean(axis=0)
            sigma_array = std_arrays.mean(axis=0)

            if self.do_scale:
                z_min_arrays = np.stack(group_df["row_z_min_array"], axis=0)
                z_max_arrays = np.stack(group_df["row_z_max_array"], axis=0)
                z_min_array = z_min_arrays.mean(axis=0)
                z_max_array = z_max_arrays.mean(axis=0)
            else:
                z_min_array = None
                z_max_array = None

            grouped_stats[tuple(group_vals)] = (
                mu_array,
                sigma_array,
                z_min_array,
                z_max_array,
            )

        self.group_stats = grouped_stats

    def create_training_dataset(self):
        data_tuples = []
        for ctx_tuple, (
            mu_arr,
            sigma_arr,
            zmin_arr,
            zmax_arr,
        ) in self.group_stats.items():
            # If scale=False, zmin_arr, zmax_arr could be None. We'll store zeros or something
            if self.do_scale and zmin_arr is not None and zmax_arr is not None:
                data_tuples.append((ctx_tuple, mu_arr, sigma_arr, zmin_arr, zmax_arr))
            else:
                data_tuples.append((ctx_tuple, mu_arr, sigma_arr, None, None))

        class _TrainSet(Dataset):
            def __init__(self, samples, context_vars, n_dims, do_scale):
                super().__init__()
                self.samples = samples
                self.context_vars = context_vars
                self.n_dims = n_dims
                self.do_scale = do_scale

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                ctx_tuple, mu_arr, sigma_arr, zmin_arr, zmax_arr = self.samples[idx]
                cat_vars_dict = {}
                for i, var_name in enumerate(self.context_vars):
                    cat_vars_dict[var_name] = torch.tensor(
                        ctx_tuple[i], dtype=torch.long
                    )
                mu_t = torch.from_numpy(mu_arr).float()
                sigma_t = torch.from_numpy(sigma_arr).float()

                if self.do_scale and (zmin_arr is not None) and (zmax_arr is not None):
                    zmin_t = torch.from_numpy(zmin_arr).float()
                    zmax_t = torch.from_numpy(zmax_arr).float()
                else:
                    # If scale=False or they're None
                    zmin_t = None
                    zmax_t = None

                return cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t

        return _TrainSet(self.samples, self.context_vars, self.n_dims, self.do_scale)

    def train_normalizer(self):
        ds = self.create_training_dataset()
        loader = DataLoader(ds, batch_size=self.normalizer_cfg.batch_size, shuffle=True)

        self.optim = torch.optim.Adam(
            self.normalizer_model.parameters(), lr=self.normalizer_cfg.lr
        )
        self.normalizer_model.train()

        for epoch in range(self.normalizer_cfg.n_epochs):
            epoch_loss = 0.0
            for batch in loader:
                cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch

                # Move to device
                for var_name in cat_vars_dict:
                    cat_vars_dict[var_name] = cat_vars_dict[var_name].to(self.device)
                mu_t = mu_t.to(self.device)
                sigma_t = sigma_t.to(self.device)

                if self.do_scale and (zmin_t is not None) and (zmax_t is not None):
                    zmin_t = zmin_t.to(self.device)
                    zmax_t = zmax_t.to(self.device)

                self.optim.zero_grad()

                pred_mu, pred_sigma, pred_z_min, pred_z_max = self.normalizer_model(
                    cat_vars_dict
                )
                # pred_mu, pred_sigma always shape (bs, n_dims)
                # pred_z_min, pred_z_max either shape (bs, n_dims) or None

                loss_mu = F.mse_loss(pred_mu, mu_t)
                loss_sigma = F.mse_loss(pred_sigma, sigma_t)
                total_loss = loss_mu + loss_sigma

                if (
                    self.do_scale
                    and pred_z_min is not None
                    and pred_z_max is not None
                    and zmin_t is not None
                    and zmax_t is not None
                ):
                    loss_z_min = F.mse_loss(pred_z_min, zmin_t)
                    loss_z_max = F.mse_loss(pred_z_max, zmax_t)
                    total_loss += loss_z_min + loss_z_max

                total_loss.backward()
                self.optim.step()
                epoch_loss += total_loss.item()

            print(
                f"Epoch {epoch+1}/{self.normalizer_cfg.n_epochs}: Loss={epoch_loss:.4f}"
            )

    def transform(self, use_model: bool = False) -> pd.DataFrame:
        df = self.dataset.data.copy()
        for i, row in df.iterrows():
            cat_vars_dict = {
                vn: torch.tensor(
                    row[vn], dtype=torch.long, device=self.device
                ).unsqueeze(0)
                for vn in self.context_vars
            }

            if use_model:
                with torch.no_grad():
                    pmu, psigma, pzmin, pzmax = self.normalizer_model(cat_vars_dict)
                mu_arr = pmu[0].cpu().numpy()
                sigma_arr = psigma[0].cpu().numpy()
                if self.do_scale and pzmin is not None and pzmax is not None:
                    zmin_arr = pzmin[0].cpu().numpy()
                    zmax_arr = pzmax[0].cpu().numpy()
                else:
                    zmin_arr = None
                    zmax_arr = None
            else:
                ctx_tuple = tuple(row[vn] for vn in self.context_vars)
                mu_arr, sigma_arr, zmin_arr, zmax_arr = self.group_stats[ctx_tuple]

            for d, col_name in enumerate(self.time_series_cols):
                arr = np.array(row[col_name], dtype=np.float32)
                z = (arr - mu_arr[d]) / (sigma_arr[d] + 1e-8)
                if self.do_scale and (zmin_arr is not None) and (zmax_arr is not None):
                    rng = (zmax_arr[d] - zmin_arr[d]) + 1e-8
                    z = (z - zmin_arr[d]) / rng
                df.at[i, col_name] = z
        return df

    def inverse_transform(
        self, df: pd.DataFrame, use_model: bool = True
    ) -> pd.DataFrame:
        for i, row in df.iterrows():
            cat_vars_dict = {
                vn: torch.tensor(
                    row[vn], dtype=torch.long, device=self.device
                ).unsqueeze(0)
                for vn in self.context_vars
            }

            if use_model:
                with torch.no_grad():
                    pmu, psigma, pzmin, pzmax = self.normalizer_model(cat_vars_dict)
                mu_arr = pmu[0].cpu().numpy()
                sigma_arr = psigma[0].cpu().numpy()
                if self.do_scale and pzmin is not None and pzmax is not None:
                    zmin_arr = pzmin[0].cpu().numpy()
                    zmax_arr = pzmax[0].cpu().numpy()
                else:
                    zmin_arr = None
                    zmax_arr = None
            else:
                ctx_tuple = tuple(row[vn] for vn in self.context_vars)
                mu_arr, sigma_arr, zmin_arr, zmax_arr = self.group_stats[ctx_tuple]

            for d, col_name in enumerate(self.time_series_cols):
                z = np.array(row[col_name], dtype=np.float32)
                if self.do_scale and (zmin_arr is not None) and (zmax_arr is not None):
                    rng = (zmax_arr[d] - zmin_arr[d]) + 1e-8
                    z = z * rng + zmin_arr[d]
                arr_orig = z * (sigma_arr[d] + 1e-8) + mu_arr[d]
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
        checkpoint = torch.load(
            path, map_location=torch.device(self.normalizer_cfg.device)
        )
        self.normalizer_model.load_state_dict(checkpoint["normalizer_model_state"])
        print(f"Loaded Normalizer from {path}")

    def _init_normalizer_config(self):
        config_path = os.path.join(ROOT_DIR, "config", "normalizer", "default.yaml")
        normalizer_cfg = OmegaConf.load(config_path)
        return normalizer_cfg
