from abc import ABC, abstractmethod

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from cents.models.context import ContextModule


class BaseModel(pl.LightningModule, ABC):
    """
    Abstract base class for all Cents PyTorch-Lightning models.

    This class handles common setup:
    - stores the Hydra configuration object
    - builds a ContextModule if context variables are defined in the dataset config

    Subclasses must implement the core Lightning methods:
    `training_step`, `configure_optimizers`, and `forward`.
    """

    def __init__(self, cfg: DictConfig = None):
        """
        Initialize the base model.

        Args:
            cfg (DictConfig): Hydra configuration with at least:
                - cfg.dataset.context_vars: dict of context variable sizes
                - cfg.model.cond_emb_dim: dimensionality of context embeddings (if context_vars non-empty)

        Raises:
            ValueError: If `cfg.dataset.context_vars` is non-empty but `cfg.model.cond_emb_dim` is missing.
        """
        super().__init__()
        if cfg is not None:
            self.cfg = cfg

            if hasattr(cfg.dataset, "context_vars") and cfg.dataset.context_vars:
                emb_dim = getattr(cfg.model, "cond_emb_dim", 256)
                self.context_module = ContextModule(cfg.dataset.context_vars, emb_dim)
            else:
                self.context_module = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        LightningModule forward method.

        Subclasses must override to define the computation of one batch,
        typically returning a loss or logits for logging.

        Args:
            *args: Positional inputs (e.g., batch data).
            **kwargs: Keyword inputs (e.g., batch index).

        Returns:
            Depends on model: usually loss tensor or prediction logits.
        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Defines a single training iteration.

        Subclasses implement this to compute loss, call `self.log(...)`,
        and return a loss tensor or dict.

        Args:
            batch: One batch of data (format defined by dataset).
            batch_idx (int): Batch index in the current epoch.

        Returns:
            torch.Tensor or dict: Training loss or metrics.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Set up PyTorch optimizers and (optionally) schedulers.

        Returns:
            Optimizer, or dict/list per Lightning docs, e.g.:
            - optimizer
            - (optimizer, scheduler)
            - {'optimizer': opt, 'lr_scheduler': scheduler, 'monitor': metric}
        """
        pass


class GenerativeModel(BaseModel):
    """
    Base class for generative time-series models.

    Subclasses must implement the `generate` API in addition to Lightning methods.
    """

    @abstractmethod
    def generate(self, context_vars: dict) -> torch.Tensor:
        """
        Produce synthetic time-series conditioned on provided context.

        Args:
            context_vars (dict): Mapping from context variable names to
                torch.Tensor of shape (batch_size,) with integer codes.

        Returns:
            torch.Tensor: Generated series of shape
                (batch_size, seq_len, time_series_dims).
        """
        pass


class NormalizerModel(BaseModel):
    """
    Base class for normalization modules.

    Subclasses must implement `transform` to normalize and
    `inverse_transform` to denormalize pandas DataFrames.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to raw time-series columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with raw series columns.

        Returns:
            pd.DataFrame: DataFrame with normalized series.
        """
        pass

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert normalization to restore original scale.

        Args:
            df (pd.DataFrame): Input DataFrame with normalized series.

        Returns:
            pd.DataFrame: DataFrame with denormalized series.
        """
        pass
