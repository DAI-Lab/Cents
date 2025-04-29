# endata/trainer.py
import inspect
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from endata.data_generator import DataGenerator
from endata.datasets.timeseries_dataset import TimeSeriesDataset
from endata.eval.eval import Evaluator
from endata.models.acgan import ACGAN
from endata.models.diffusion_ts import Diffusion_TS
from endata.models.normalizer import Normalizer
from endata.utils.utils import get_default_normalizer_config, get_device

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"


class Trainer:
    """
    Facade that hides Hydra + Lightning while supporting ACGAN, Diffusion_TS
    and the parametric Normalizer.

    Parameters
    ----------
    model        : str        # "acgan" | "diffusion_ts" | "normalizer"
    dataset      : TimeSeriesDataset | None
    cfg          : DictConfig | None   # override *whole* cfg (advanced)
    overrides    : list[str] | None    # Hydra-style dot-list overrides
    """

    _MODEL_REGISTRY = {
        "acgan": ACGAN,
        "diffusion_ts": Diffusion_TS,
        "normalizer": Normalizer,
    }

    def __init__(
        self,
        model: str,
        dataset: Optional[TimeSeriesDataset] = None,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
    ):
        if model not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model}'. " f"Available: {list(self._MODEL_REGISTRY)}"
            )

        if model != "normalizer" and dataset is None:
            raise ValueError(f"Model '{model}' requires a TimeSeriesDataset.")

        if model == "normalizer" and dataset is None:
            raise ValueError("Normalizer training needs the raw dataset object.")

        self.model_key = model
        self.dataset = dataset
        self.cfg = cfg or self._compose_cfg(overrides or [])
        self.device = get_device(self.cfg.device)

        self.model = self._instantiate_model()
        self.pl_trainer = self._instantiate_trainer()

    def fit(self):
        """Start training. Returns self for chaining."""
        if self.model_key == "normalizer":
            self.pl_trainer.fit(self.model)
        else:
            train_loader = self.dataset.get_train_dataloader(
                batch_size=self.cfg.trainer.batch_size, shuffle=True, num_workers=4
            )
            self.pl_trainer.fit(self.model, train_loader, None)
        return self

    def get_data_generator(self) -> DataGenerator:
        """Return a ready-to-use DataGenerator bound to this trained model."""
        if self.model_key == "normalizer":
            raise RuntimeError("Normalizer is not a generative model.")

        gen = DataGenerator(
            model_name=self.model_key,
            device=self.device,
            cfg=self.cfg,
            model=self.model.eval(),
            normalizer=getattr(self.dataset, "_normalizer", None),
        )

        gen.set_dataset_spec(
            dataset_cfg=self.dataset.cfg,
            ctx_codes=self.dataset.get_context_var_codes(),
        )
        return gen

    def evaluate(self, **kwargs):
        evaluator = Evaluator(self.cfg, self.dataset)
        return evaluator.evaluate_model(model=self.model, **kwargs)

    def _compose_cfg(self, ov: List[str]) -> DictConfig:
        """
        1. load root config.yaml
        2. inject `model=<key>` and `trainer=<key>` defaults
        3. merge dataset-specific cfg (already present inside dataset)
        4. apply CLI overrides if any
        """
        base_ov = [f"model={self.model_key}", f"trainer={self.model_key}"]
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            cfg = compose(config_name="config", overrides=base_ov + ov)
        if self.dataset is not None:
            cfg.dataset = self.dataset.cfg
        return cfg

    def _instantiate_model(self):
        ModelCls = self._MODEL_REGISTRY[self.model_key]
        if self.model_key == "normalizer":
            nm_cfg = get_default_normalizer_config()
            mdl = ModelCls(
                dataset_cfg=self.cfg.dataset,
                normalizer_cfg=nm_cfg,
                dataset=self.dataset,
            )
        else:
            mdl = ModelCls(self.cfg)
        mdl.to(self.device)
        return mdl

    def _instantiate_trainer(self) -> pl.Trainer:
        tc = self.cfg.trainer
        ckp_cfg = tc.get("checkpoint", {})
        callbacks = []
        if "monitor" in ckp_cfg:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckp_cfg.get("dir", "checkpoints"),
                    monitor=ckp_cfg.monitor,
                    mode=ckp_cfg.mode,
                    save_top_k=ckp_cfg.save_top_k,
                )
            )
        else:
            every_n = ckp_cfg.get("every_n_train_steps", tc.get("save_cycle", 0))
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckp_cfg.get("dir", "checkpoints"),
                    filename="{step}",
                    every_n_train_steps=every_n,
                    save_last=ckp_cfg.get("save_last", True),
                    save_on_train_epoch_end=False,
                )
            )

        logger = None
        if "wandb" in self.cfg and self.cfg.wandb.enabled:
            logger = WandbLogger(
                project=self.cfg.wandb.project or "EnData",
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                save_dir=self.cfg.run_dir,
            )
        else:
            logger = CSVLogger(save_dir=self.cfg.run_dir, name="logs")

        return pl.Trainer(
            max_epochs=tc.max_epochs,
            accelerator=tc.accelerator,
            devices=tc.devices,
            precision=tc.precision,
            log_every_n_steps=tc.get("log_every_n_steps", 1),
            accumulate_grad_batches=tc.get("gradient_accumulate_every", 1),
            callbacks=callbacks,
            logger=logger,
            default_root_dir=self.cfg.run_dir,
        )
