from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cents.data_generator import DataGenerator
from cents.datasets.timeseries_dataset import TimeSeriesDataset
from cents.eval.eval import Evaluator
from cents.models.registry import get_model_cls
from cents.utils.utils import get_normalizer_training_config

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"


class Trainer:
    """
    Facade for training and evaluating generative time-series models.

    Supports ACGAN, Diffusion_TS and Normalizer via PyTorch Lightning and Hydra.

    Attributes:
        model_type: Identifier of the model to train/evaluate.
        dataset: TimeSeriesDataset used for training and evaluation.
        cfg: Hydra configuration object.
        model: Instantiated model object.
        pl_trainer: PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        model_type: str,
        dataset: Optional[TimeSeriesDataset] = None,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            model_type: Key of the model ("acgan", "diffusion_ts", or "normalizer").
            dataset: Dataset object required for generative models; optional for normalizer.
            cfg: Full OmegaConf DictConfig; if None, composed via Hydra.
            overrides: List of Hydra override strings.

        Raises:
            ValueError: If model_type is unknown or dataset requirements are not met.
        """
        try:
            get_model_cls(model_type)
        except ValueError:
            raise ValueError(f"Unknown model '{model_type}'")

        if model_type != "normalizer" and dataset is None:
            raise ValueError(f"Model '{model_type}' requires a TimeSeriesDataset.")

        if model_type == "normalizer" and dataset is None:
            raise ValueError("Normalizer training needs the raw dataset object.")

        self.model_type = model_type
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
        if self.model_type == "normalizer":
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
        if self.model_type == "normalizer":
            raise RuntimeError("Normalizer is not a generative model.")

        device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

        gen = DataGenerator(
            model_name=self.model_type,
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
        base_ov = [f"model={self.model_type}", f"trainer={self.model_type}"]
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            cfg = compose(config_name="config", overrides=base_ov + ov)
        if self.dataset is not None:
            cfg.dataset = OmegaConf.create(
                OmegaConf.to_container(self.dataset.cfg, resolve=True)
            )
        return cfg

    def _instantiate_model(self):
        """
        Instantiate the model class from the registry based on model_type.
        """
        ModelCls = get_model_cls(self.model_type)
        if self.model_type == "normalizer":
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
                    f"{self.cfg.dataset.name}_{self.model_type}"
                    f"_dim{self.cfg.dataset.time_series_dims}"
                ),
                save_last=tc.checkpoint.save_last,
                save_on_train_epoch_end=True,
            )
        )
        callbacks.append(EvalAfterTraining(self.cfg, self.dataset))
        logger = False
        if getattr(self.cfg, "wandb", None) and self.cfg.wandb.enabled:
            logger = WandbLogger(
                project=self.cfg.wandb.project or "cents",
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


class EvalAfterTraining(Callback):
    """Run full evaluator at the *end* of training and log metrics to W&B."""

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def on_train_end(self, trainer, pl_module):
        if not self.cfg.trainer.get("eval_after_training", False):
            return

        evaluator = Evaluator(self.cfg, self.dataset)
        results = evaluator.evaluate_model(model=pl_module)

        run = getattr(trainer.logger, "experiment", None)
        if run is not None:
            run.log(results["metrics"])
