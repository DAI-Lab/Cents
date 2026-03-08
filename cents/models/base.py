from abc import ABC, abstractmethod

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from cents.models.context import MLPContextModule, SepMLPContextModule  # Import to trigger registration
from cents.models.context_registry import get_context_module_cls
from cents.utils.utils import get_context_config


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
                # Get context module type from context config
                context_cfg = get_context_config()
                static_module_type = context_cfg.static_context.type
                dynamic_module_type = getattr(context_cfg.dynamic_context, "type", None)
                
                # Separate static and dynamic context variables
                continuous_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "continuous"]
                categorical_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "categorical"]
                dynamic_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "time_series"]
                
                static_context_vars = categorical_vars + continuous_vars
                self.dynamic_context_vars = dynamic_vars
                
                # Create static context module (for categorical + continuous)
                self.static_context_module = None
                if static_context_vars:
                    StaticContextModuleCls = get_context_module_cls(static_module_type)
                    static_context_vars_dict = {
                        k: v for k, v in cfg.dataset.context_vars.items() 
                        if k in static_context_vars
                    }
                    print(static_context_vars_dict)
                    self.static_context_module = StaticContextModuleCls(
                        static_context_vars_dict,
                        emb_dim,
                    )
                
                # Create dynamic context module (for time_series)
                self.dynamic_context_module = None
                if self.dynamic_context_vars and dynamic_module_type is not None:
                    num_ts_steps = getattr(cfg.dataset, "num_ts_steps", None)
                    DynamicContextModuleCls = get_context_module_cls("dynamic", dynamic_module_type)
                    dynamic_context_vars_dict = {
                        k: v for k, v in cfg.dataset.context_vars.items() 
                        if k in self.dynamic_context_vars
                    }
                    seq_len = getattr(cfg.dataset, "seq_len", None)
                    if seq_len is None:
                        raise ValueError("seq_len must be specified in cfg.dataset for dynamic context modules")
                    self.dynamic_context_module = DynamicContextModuleCls(
                        dynamic_context_vars_dict,
                        emb_dim,
                        seq_len=seq_len if num_ts_steps is None else num_ts_steps,
                    )
                
                # Determine embedding dimension
                if self.static_context_module is not None:
                    self.embedding_dim = self.static_context_module.embedding_dim
                elif self.dynamic_context_module is not None:
                    self.embedding_dim = self.dynamic_context_module.embedding_dim
                else:
                    raise ValueError("At least one of static_context_module or dynamic_context_module must be provided")

                # combine_mlp is only needed when the dynamic module returns a pooled
                # vector (returns_sequence=False) that must be fused with the static
                # embedding before AdaLN conditioning.  When returns_sequence=True the
                # dynamic context is routed to cross-attention inside the backbone and
                # must NOT be merged with the static embedding here.
                dyn_returns_seq = getattr(self.dynamic_context_module, "returns_sequence", False) \
                    if self.dynamic_context_module is not None else False

                if (self.static_context_module is not None
                        and self.dynamic_context_module is not None
                        and not dyn_returns_seq):
                    combined_dim = self.static_context_module.embedding_dim + self.dynamic_context_module.embedding_dim
                    self.combine_mlp = nn.Sequential(
                        nn.Linear(combined_dim, self.embedding_dim),
                        nn.ReLU(),
                    )
                else:
                    self.combine_mlp = None
                
                # For backward compatibility, expose static_context_module as context_module
                # (but subclasses should use static_context_module and dynamic_context_module directly)
                self.context_module = self.static_context_module
            else:
                self.static_context_module = None
                self.dynamic_context_module = None
                self.context_module = None
                self.combine_mlp = None
                self.dynamic_context_vars = []
                self.embedding_dim = getattr(cfg.model, "cond_emb_dim", 256) if cfg is not None else 256

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
