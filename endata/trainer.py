import inspect
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from endata.data_generator import DataGenerator
from endata.datasets.timeseries_dataset import TimeSeriesDataset
from endata.eval.eval import Evaluator
from endata.models.registry import get_model_cls
from endata.utils.utils import get_normalizer_training_config

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"


class Trainer:
    """
    Facade for training and evaluating generative time-series models.

    Supports ACGAN, Diffusion_TS and Normalizer via PyTorch Lightning and Hydra.

    Attributes:
        model_key: Identifier of the model to train/evaluate.
        dataset: TimeSeriesDataset used for training and evaluation.
        cfg: Hydra configuration object.
        model: Instantiated model object.
        pl_trainer: PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        model_name: str,
        dataset: Optional[TimeSeriesDataset] = None,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            model_name: Key of the model ("acgan", "diffusion_ts", or "normalizer").
            dataset: Dataset object required for generative models; optional for normalizer.
            cfg: Full OmegaConf DictConfig; if None, composed via Hydra.
            overrides: List of Hydra override strings.

        Raises:
            ValueError: If model_name is unknown or dataset requirements are not met.
        """
        try:
            get_model_cls(model_name)
        except ValueError:
            raise ValueError(f"Unknown model '{model_name}'")

        if model_name != "normalizer" and dataset is None:
            raise ValueError(f"Model '{model_name}' requires a TimeSeriesDataset.")

        if model_name == "normalizer" and dataset is None:
            raise ValueError("Normalizer training needs the raw dataset object.")

        self.model_key = model_name
        self.dataset = dataset
        self.cfg = cfg or self._compose_cfg(overrides or [])

        self.model = self._instantiate_model()
        self.pl_trainer = self._instantiate_trainer()

    def fit(self) -> "Trainer":
        """
        Start training.

        Returns:
            Self, to allow method chaining.
        """
        if self.model_key == "normalizer":
            self.pl_trainer.fit(self.model)
        else:
            train_loader = self.dataset.get_train_dataloader(
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=4,
            )
            self.pl_trainer.fit(self.model, train_loader, None)
        return self

    def get_data_generator(self) -> DataGenerator:
        """
        Create a DataGenerator for sampling from the trained generative model.

        Returns:
            DataGenerator bound to the trained model and dataset.

        Raises:
            RuntimeError: If called for the normalizer model (non-generative).
        """
        if self.model_key == "normalizer":
            raise RuntimeError("Normalizer is not a generative model.")

        device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

        gen = DataGenerator(
            model_name=self.model_key,
            device=device,
            cfg=self.cfg,
            model=self.model.eval(),
            normalizer=getattr(self.dataset, "_normalizer", None),
        )

        gen.set_dataset_spec(
            dataset_cfg=self.dataset.cfg,
            ctx_codes=self.dataset.get_context_var_codes(),
        )
        return gen

    def evaluate(self, **kwargs) -> Dict:
        """
        Run evaluation of the trained model using Evaluator.

        Args:
            **kwargs: Passed to Evaluator.evaluate_model (e.g. user_id).

        Returns:
            Dictionary of evaluation results.
        """
        evaluator = Evaluator(self.cfg, self.dataset)
        return evaluator.evaluate_model(model=self.model, **kwargs)

    def _compose_cfg(self, ov: List[str]) -> DictConfig:
        """
        Compose the full Hydra configuration by merging defaults,
        dataset-specific config, and any user overrides.

        Args:
            ov: List of Hydra-style overrides.

        Returns:
            OmegaConf DictConfig.
        """
        base_ov = [f"model={self.model_key}", f"trainer={self.model_key}"]
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            cfg = compose(config_name="config", overrides=base_ov + ov)
        if self.dataset is not None:
            cfg.dataset = self.dataset.cfg
        return cfg

    def _instantiate_model(self):
        """
        Instantiate the model class from the registry based on model_key.
        """
        ModelCls = get_model_cls(self.model_key)
        if self.model_key == "normalizer":
            nm_cfg = get_normalizer_training_config()
            return ModelCls(
                dataset_cfg=self.cfg.dataset,
                normalizer_training_cfg=nm_cfg,
                dataset=self.dataset,
            )
        return ModelCls(self.cfg)

    def _instantiate_trainer(self) -> pl.Trainer:
        """
        Build a PyTorch Lightning Trainer with ModelCheckpoint and loggers.

        Returns:
            Configured pl.Trainer instance.
        """
        tc = self.cfg.trainer
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.cfg.run_dir,
                filename=(
                    f"{self.cfg.dataset.name}_{self.model_key}"
                    f"_dim{self.cfg.dataset.time_series_dims}"
                ),
                save_last=tc.checkpoint.save_last,
                save_on_train_epoch_end=True,
            )
        )

        logger = False
        if getattr(self.cfg, "wandb", None) and self.cfg.wandb.enabled:
            logger = WandbLogger(
                project=self.cfg.wandb.project or "EnData",
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                save_dir=self.cfg.run_dir,
            )

        return pl.Trainer(
            max_epochs=tc.max_epochs,
            accelerator=tc.accelerator,
            strategy=tc.strategy,
            devices=tc.devices,
            precision=tc.precision,
            log_every_n_steps=tc.get("log_every_n_steps", 1),
            accumulate_grad_batches=tc.get("gradient_accumulate_every", 1),
            callbacks=callbacks,
            logger=logger,
            default_root_dir=self.cfg.run_dir,
        )
