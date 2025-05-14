from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cents.datasets.utils import split_timeseries
from cents.models.base import NormalizerModel
from cents.models.context import ContextModule
from cents.models.registry import register_model


class _StatsHead(nn.Module):
    """
    Head module predicting summary statistics (mean, std, and optionally min/max z-scores) from context embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        time_series_dims: int,
        do_scale: bool,
        n_layers: int = 3,
    ):
        """
        Initializes the statistics head network.

        Args:
            embedding_dim: Dimensionality of the input context embedding.
            hidden_dim: Number of units in each hidden layer.
            time_series_dims: Number of dimensions in the original time series.
            do_scale: Whether to predict scaling min/max parameters.
            n_layers: Number of hidden linear layers before the output.
        """
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
        """
        Forward pass to compute predicted statistics.

        Args:
            z: Context embedding tensor of shape (batch_size, embedding_dim).

        Returns:
            pred_mu: Predicted means, shape (batch_size, time_series_dims).
            pred_sigma: Predicted standard deviations, shape (batch_size, time_series_dims).
            pred_z_min: Predicted min z-scores, or None if do_scale=False.
            pred_z_max: Predicted max z-scores, or None if do_scale=False.
        """
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
    """
    Wrapper module combining a context embedding and stats head for normalization.
    """

    def __init__(
        self,
        cond_module: nn.Module,
        hidden_dim: int = 512,
        time_series_dims: int = 2,
        do_scale: bool = True,
    ):
        """
        Args:
            cond_module: ContextModule instance for embedding context variables.
            hidden_dim: Hidden dimension size for the stats head.
            time_series_dims: Number of time series dimensions.
            do_scale: Whether to include scaling predictions.
        """
        super().__init__()
        self.cond_module = cond_module
        self.embedding_dim = cond_module.embedding_dim
        self.stats_head = _StatsHead(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            time_series_dims=time_series_dims,
            do_scale=do_scale,
        )

    def forward(self, cat_vars_dict: dict):
        """
        Compute normalization parameters from categorical context.

        Args:
            cat_vars_dict: Mapping of context variable names to label tensors.

        Returns:
            Tuple of (pred_mu, pred_sigma, pred_z_min, pred_z_max).
        """
        embedding, _ = self.cond_module(cat_vars_dict)
        return self.stats_head(embedding)


@register_model("normalizer")
class Normalizer(NormalizerModel):
    """
    Learns group-wise normalization parameters (mean, std, optional min/max) for time series by context.
    """

    def __init__(
        self,
        dataset_cfg,
        normalizer_training_cfg,
        dataset,
    ):
        """
        Initializes the Normalizer training module.

        Args:
            dataset_cfg: OmegaConf dataset config (provides context_vars, columns).
            normalizer_training_cfg: Config for normalizer training (lr, batch_size).
            dataset: Instance of TimeSeriesDataset containing data DataFrame.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset_cfg = dataset_cfg
        self.normalizer_training_cfg = normalizer_training_cfg
        self.dataset = dataset

        self.context_vars = list(dataset_cfg.context_vars.keys())
        self.time_series_cols = dataset_cfg.time_series_columns[
            : dataset_cfg.time_series_dims
        ]
        self.time_series_dims = dataset_cfg.time_series_dims
        self.do_scale = dataset_cfg.scale

        self.context_module = ContextModule(
            dataset_cfg.context_vars,
            256,
        )

        self.normalizer_model = _NormalizerModule(
            cond_module=self.context_module,
            hidden_dim=512,
            time_series_dims=self.time_series_dims,
            do_scale=self.do_scale,
        )

        # Will be populated in setup()
        self.group_stats = {}

    def setup(self, stage: Optional[str] = None):
        """
        Lightning hook: compute group statistics before training.
        """
        self.group_stats = self._compute_group_stats()

    def forward(self, cat_vars_dict: dict):
        """
        Predict normalization parameters for a batch of categorical contexts.

        Args:
            cat_vars_dict: Mapping of context variable names to label tensors.

        Returns:
            Tuple of (pred_mu, pred_sigma, pred_z_min, pred_z_max).
        """
        return self.normalizer_model(cat_vars_dict)

    def training_step(self, batch, batch_idx: int):
        """
        Training step: regress predicted stats against true group stats.

        Args:
            batch: Tuple of (cat_vars_dict, mu, sigma, zmin, zmax).
            batch_idx: Batch index.

        Returns:
            loss tensor.
        """
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
        """
        Configure optimizer for normalizer training.

        Returns:
            Adam optimizer instance.
        """
        return torch.optim.Adam(self.parameters(), lr=self.normalizer_training_cfg.lr)

    def train_dataloader(self):
        """
        Returns a DataLoader over per-group statistics samples.
        """
        ds = self._create_training_dataset()
        return DataLoader(
            ds,
            batch_size=self.normalizer_training_cfg.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def _compute_group_stats(self) -> dict:
        """
        Compute per-group (context combination) statistics from raw data.

        Returns:
            Mapping from context tuple to (mu_array, std_array, zmin_array, zmax_array).
        """
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

    def _create_training_dataset(self) -> Dataset:
        """
        Build an internal Dataset yielding true stats for each context group.

        Returns:
            PyTorch Dataset of samples (cat_vars_dict, mu, sigma, zmin, zmax).
        """
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
            """
            Adapter Dataset to wrap group_stats tuples for DataLoader.
            """

            def __init__(self, samples, context_vars, do_scale):
                self.samples = samples
                self.context_vars = context_vars
                self.do_scale = do_scale

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int):
                """
                Returns one training sample.

                Args:
                    idx: Index of the sample.

                Returns:
                    cat_vars_dict: Tensor dict of context labels.
                    mu_t: True mean tensor.
                    sigma_t: True std tensor.
                    zmin_t: True min z-score tensor or None.
                    zmax_t: True max z-score tensor or None.
                """
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame of time series using learned parameters.

        Pads or splits if needed, then applies z-score and min-max scaling.

        Args:
            df: Input DataFrame with raw time series columns.

        Returns:
            DataFrame with normalized series in same columns.
        """
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = split_timeseries(df, self.time_series_cols)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.transform expects data in split format with columns "
            f"{self.time_series_cols}."
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

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalize a DataFrame of z-scored series back to original scale.

        Args:
            df: DataFrame with normalized series columns.

        Returns:
            DataFrame with denormalized series.
        """
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = split_timeseries(df, self.time_series_cols)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.inverse_transform expects split format with columns "
            f"{self.time_series_cols}."
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
