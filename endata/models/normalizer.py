import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from endata.models.context import ContextModule


class _StatsHead(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, time_series_dims, do_scale, n_layers=3
    ):
        super().__init__()
        self.time_series_dims = time_series_dims
        self.do_scale = do_scale
        out_dim = 4 * time_series_dims if do_scale else 2 * time_series_dims
        layers = []
        in_dim = embedding_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        batch_size = out.size(0)
        if self.do_scale:
            out = out.view(batch_size, 4, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = out[:, 2, :]
            pred_z_max = out[:, 3, :]
        else:
            out = out.view(batch_size, 2, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = None
            pred_z_max = None
        pred_sigma = torch.exp(pred_log_sigma)
        return pred_mu, pred_sigma, pred_z_min, pred_z_max


class _NormalizerModule(nn.Module):
    def __init__(
        self,
        cond_module: nn.Module,
        hidden_dim: int,
        time_series_dims: int,
        do_scale: bool,
    ):
        super().__init__()
        self.cond_module = cond_module
        self.embedding_dim = cond_module.embedding_dim
        self.stats_head = _StatsHead(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            time_series_dims=time_series_dims,
            do_scale=do_scale,
        )

    def forward(self, cat_vars_dict):
        embedding, _ = self.cond_module(cat_vars_dict)
        return self.stats_head(embedding)


class Normalizer(pl.LightningModule):
    def __init__(self, dataset_cfg, normalizer_cfg, dataset):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset_cfg = dataset_cfg
        self.normalizer_cfg = normalizer_cfg
        self.dataset = dataset

        self.context_vars = list(dataset_cfg.context_vars.keys())
        self.time_series_cols = dataset_cfg.time_series_columns
        self.time_series_dims = dataset_cfg.time_series_dims
        self.do_scale = dataset_cfg.scale

        self.context_module = ContextModule(
            dataset_cfg.context_vars,
            normalizer_cfg.embedding_dim,
        )

        self.normalizer_model = _NormalizerModule(
            cond_module=self.context_module,
            hidden_dim=normalizer_cfg.hidden_dim,
            time_series_dims=self.time_series_dims,
            do_scale=self.do_scale,
        )

        self.group_stats = {}

    def setup(self, stage=None):
        self.group_stats = self._compute_group_stats()

    def forward(self, cat_vars_dict):
        return self.normalizer_model(cat_vars_dict)

    def training_step(self, batch, batch_idx):
        cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch
        pred_mu, pred_sigma, pred_z_min, pred_z_max = self(cat_vars_dict)

        loss_mu = F.mse_loss(pred_mu, mu_t)
        loss_sigma = F.mse_loss(pred_sigma, sigma_t)
        total_loss = loss_mu + loss_sigma

        if self.do_scale:
            total_loss += F.mse_loss(pred_z_min, zmin_t) + F.mse_loss(
                pred_z_max, zmax_t
            )

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.normalizer_cfg.lr)

    def train_dataloader(self):
        ds = self._create_training_dataset()
        return DataLoader(ds, batch_size=self.normalizer_cfg.batch_size, shuffle=True)

    def _compute_group_stats(self):
        df = self.dataset.data.copy()
        grouped_stats = {}
        for group_vals, group_df in df.groupby(self.context_vars):
            dimension_points = [[] for _ in range(self.time_series_dims)]
            for _, row in group_df.iterrows():
                for d, col_name in enumerate(self.time_series_cols):
                    arr = np.array(row[col_name], dtype=np.float32).flatten()
                    dimension_points[d].append(arr)
            dimension_points = [np.concatenate(d, axis=0) for d in dimension_points]
            mu_array = np.array(
                [pts.mean() for pts in dimension_points], dtype=np.float32
            )
            std_array = np.array(
                [pts.std() + 1e-8 for pts in dimension_points], dtype=np.float32
            )

            if self.do_scale:
                z_min_array = np.array(
                    [
                        (pts - mu).min() / std
                        for pts, mu, std in zip(dimension_points, mu_array, std_array)
                    ],
                    dtype=np.float32,
                )
                z_max_array = np.array(
                    [
                        (pts - mu).max() / std
                        for pts, mu, std in zip(dimension_points, mu_array, std_array)
                    ],
                    dtype=np.float32,
                )
            else:
                z_min_array = z_max_array = None

            grouped_stats[tuple(group_vals)] = (
                mu_array,
                std_array,
                z_min_array,
                z_max_array,
            )
        return grouped_stats

    def _create_training_dataset(self):
        data_tuples = [
            (ctx_tuple, mu_arr, sigma_arr, zmin_arr, zmax_arr)
            for ctx_tuple, (
                mu_arr,
                sigma_arr,
                zmin_arr,
                zmax_arr,
            ) in self.group_stats.items()
        ]

        class _TrainSet(Dataset):
            def __init__(self, samples, context_vars, do_scale):
                self.samples = samples
                self.context_vars = context_vars
                self.do_scale = do_scale

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                ctx_tuple, mu_arr, sigma_arr, zmin_arr, zmax_arr = self.samples[idx]
                cat_vars_dict = {
                    var_name: torch.tensor(ctx_tuple[i], dtype=torch.long)
                    for i, var_name in enumerate(self.context_vars)
                }
                mu_t = torch.from_numpy(mu_arr).float()
                sigma_t = torch.from_numpy(sigma_arr).float()
                zmin_t = torch.from_numpy(zmin_arr).float() if self.do_scale else None
                zmax_t = torch.from_numpy(zmax_arr).float() if self.do_scale else None
                return cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t

        return _TrainSet(data_tuples, self.context_vars, self.do_scale)

    def transform(self, df: pd.DataFrame):
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = self.dataset.split_timeseries(df)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.transform expects the DataFrame to ALREADY be in **split** "
            f"format with per-dimension columns {self.time_series_cols}. "
            "Call the dataset’s `split_timeseries(df)` helper first."
        )

        df_out = df.copy()
        self.eval()
        with torch.no_grad():
            for i, row in df_out.iterrows():
                ctx = {
                    v: torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                    for v in self.context_vars
                }
                mu, sigma, zmin, zmax = self(ctx)
                mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    arr = np.asarray(row[col], dtype=np.float32)
                    z = (arr - mu[d]) / (sigma[d] + 1e-8)
                    if self.do_scale:
                        zmin_, zmax_ = zmin[0, d].item(), zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        z = (z - zmin_) / rng
                    df_out.at[i, col] = z
        return df_out

    def inverse_transform(self, df: pd.DataFrame):
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = self.dataset.split_timeseries(df)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.inverse_transform expects the DataFrame to ALREADY be in "
            f"split format with per-dimension columns {self.time_series_cols}. "
            "Call the dataset’s `split_timeseries(df)` helper first."
        )

        df_out = df.copy()
        self.eval()
        with torch.no_grad():
            for i, row in df_out.iterrows():
                ctx = {
                    v: torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                    for v in self.context_vars
                }
                mu, sigma, zmin, zmax = self(ctx)
                mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    z = np.asarray(row[col], dtype=np.float32)
                    if self.do_scale:
                        zmin_, zmax_ = zmin[0, d].item(), zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        z = z * rng + zmin_
                    arr = z * (sigma[d] + 1e-8) + mu[d]
                    df_out.at[i, col] = arr
        return df_out
