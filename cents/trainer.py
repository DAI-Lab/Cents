import csv
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import wandb
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cents.data_generator import DataGenerator
from cents.datasets.timeseries_dataset import TimeSeriesDataset
from cents.eval.eval import Evaluator
from cents.models.registry import get_model_cls
from cents.utils.utils import get_normalizer_training_config
from cents.utils.config_loader import load_yaml, apply_overrides

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

    def fit(self, ckpt_path: Optional[str] = None) -> "Trainer":
        """
        Start training.

        Args:
            ckpt_path: Optional path to checkpoint file (.ckpt) to resume training from.
                      If provided, training will resume from this checkpoint.

        Returns:
            Self, to allow method chaining.
        """
        if self.model_type == "normalizer":
            self.pl_trainer.fit(self.model, ckpt_path=ckpt_path)
        else:
            train_loader = self.dataset.get_train_dataloader(
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=4,  # Maximum for 7.5GB/10GB GPU usage
                persistent_workers=True,
            )
            print(f"[Cents] Training model on {len(train_loader)} batches")
            self.pl_trainer.fit(self.model, train_loader, None, ckpt_path=ckpt_path)
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
        Compose configuration by loading YAMLs and applying overrides.

        Structure:
            cfg.model   <- config/model/{model_type}.yaml
            cfg.trainer <- config/trainer/{model_type}.yaml
            cfg.dataset <- provided dataset.cfg (if any)
        """
        model_cfg = load_yaml(CONF_DIR / "model" / f"{self.model_type}.yaml")
        trainer_cfg = load_yaml(CONF_DIR / "trainer" / f"{self.model_type}.yaml")

        cfg = OmegaConf.create({})
        cfg.model = model_cfg
        cfg.trainer = trainer_cfg

        if self.dataset is not None:
            cfg.dataset = OmegaConf.create(
                OmegaConf.to_container(self.dataset.cfg, resolve=True)
            )

        cfg = apply_overrides(cfg, ov)

        # Ensure required top-level fields exist without Hydra
        if not hasattr(cfg, "device"):
            cfg.device = "auto"
        if not hasattr(cfg, "job_name"):
            ds_name = getattr(cfg, "dataset", {}).get("name", "dataset") if isinstance(getattr(cfg, "dataset", {}), dict) else getattr(cfg.dataset, "name", "dataset")
            ds_group = getattr(cfg, "dataset", {}).get("user_group", "all") if isinstance(getattr(cfg, "dataset", {}), dict) else getattr(cfg.dataset, "user_group", "all")
            model_name = getattr(cfg.model, "name", self.model_type)
            cfg.job_name = f"{model_name}_{ds_name}_{ds_group}"
        if not hasattr(cfg, "run_dir") or not cfg.run_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cfg.run_dir = str(PKG_ROOT / "outputs" / cfg.job_name / timestamp)
        # Checkpoint dir: run_dir/checkpoints so run root stays clean
        cfg.checkpoint_dir = str(Path(cfg.run_dir) / "checkpoints")
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
        Saves checkpoints every N epochs (if configured) and always keeps last.ckpt.
        """
        tc = self.cfg.trainer
        callbacks = []

        # ---- Build descriptive base filename ----
        filename_parts = [
            self.cfg.dataset.name,
            self.model_type,
            f"dim{self.cfg.dataset.time_series_dims}",
        ]

        from cents.utils.utils import get_context_config
        context_cfg = get_context_config()

        if context_cfg.static_context.type:
            filename_parts.append(f"ctx{context_cfg.static_context.type}")

        if context_cfg.dynamic_context.type:
            filename_parts.append(f"dyn{context_cfg.dynamic_context.type}")

        if context_cfg.normalizer.stats_head_type:
            filename_parts.append(f"stats{context_cfg.normalizer.stats_head_type}")

        base_name = "_".join(filename_parts)

        # ---- Checkpoint directory ----
        checkpoint_dir = (Path(self.cfg.run_dir) / "checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---- Periodic saving config ----
        every_n = getattr(tc.checkpoint, "every_n_epochs", None)

        if every_n and every_n > 0:
            filename = f"{base_name}_epoch={{epoch:04d}}"
            every_n_epochs = every_n
            save_top_k = -1  # keep ALL periodic checkpoints
        else:
            filename = base_name
            every_n_epochs = None
            save_top_k = getattr(tc.checkpoint, "save_top_k", 1)

        print(f"Saving every {every_n_epochs} epochs (if configured)")

        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=filename,
                every_n_epochs=every_n_epochs,
                save_on_train_epoch_end=True,
                save_last=True,          # always keep last.ckpt
                save_top_k=save_top_k,
                auto_insert_metric_name=False,
            )
        )

        callbacks.append(EvalAfterTraining(self.cfg, self.dataset))

        fid_cfg = tc.get("intermediate_fid", {})
        if fid_cfg.get("enabled", False) and self.dataset is not None:
            callbacks.append(IntermediateFIDCallback(
                cfg=self.cfg,
                dataset=self.dataset,
                every_n_epochs=fid_cfg.get("every_n_epochs", 20),
                n_samples=fid_cfg.get("n_samples", 3500),
                fast_timesteps=fid_cfg.get("fast_timesteps", 50),
                top_k=fid_cfg.get("top_k", 3),
            ))

        if getattr(self.cfg, "run_dir", None):
            callbacks.append(LogLossToCsv(self.cfg.run_dir))

        # ---- Logger ----
        logger = False
        if getattr(self.cfg, "wandb", None) and self.cfg.wandb.enabled:
            wandb_id_file = Path(self.cfg.run_dir) / "wandb_run_id.txt"
            existing_run_id = wandb_id_file.read_text().strip() if wandb_id_file.exists() else None
            logger = WandbLogger(
                project=self.cfg.wandb.project or "cents",
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                save_dir=self.cfg.run_dir,
                id=existing_run_id,
                resume="must" if existing_run_id else None,
            )
            callbacks.append(_WandbRunIdSaver(self.cfg.run_dir))

        return pl.Trainer(
            max_epochs=tc.max_epochs,
            accelerator=tc.accelerator,
            strategy=tc.strategy,
            devices=tc.devices,
            precision=tc.precision,
            log_every_n_steps=tc.get("log_every_n_steps", 1),
            accumulate_grad_batches=tc.get("gradient_accumulate_every", 1),
            gradient_clip_val=tc.get("gradient_clip_val", None),
            callbacks=callbacks,
            logger=logger,
            default_root_dir=self.cfg.run_dir,
        )



class LogLossToCsv(Callback):
    """Append epoch loss values to runs/<run_name>/train_losses.csv."""

    def __init__(self, run_dir: str):
        super().__init__()
        self.run_dir = Path(run_dir)
        self._csv_path = self.run_dir / "train_losses.csv"
        # Don't rewrite the header if the file already exists (resume case)
        self._header_written = self._csv_path.exists()

    def _ensure_header(self, metric_names: List[str]) -> None:
        if self._header_written:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + metric_names)
        self._header_written = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        if not metrics:
            return
        # Filter to loss-like keys and sort for consistent column order
        loss_keys = sorted(k for k in metrics if "loss" in k.lower())
        if not loss_keys:
            return
        self._ensure_header(loss_keys)
        row = [trainer.current_epoch]
        for k in loss_keys:
            v = metrics[k]
            row.append(float(v) if hasattr(v, "item") else float(v))
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


class _WandbRunIdSaver(Callback):
    """Persist the W&B run ID to disk so --resume-from-checkpoint can continue the same run."""

    def __init__(self, run_dir: str):
        super().__init__()
        self._id_file = Path(run_dir) / "wandb_run_id.txt"

    def on_train_start(self, trainer, pl_module):  # noqa: ARG002
        run = getattr(trainer.logger, "experiment", None)
        if run is not None and not self._id_file.exists():
            self._id_file.write_text(run.id)
            print(f"[Cents] Saved W&B run ID {run.id} to {self._id_file}")


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


class IntermediateFIDCallback(Callback):
    """
    Every N epochs: generate samples, compute context-FID, save a checkpoint.

    Checkpoint retention policy:
      - Top-k epochs by FID (lower = better) are kept permanently.
      - The FID-check epochs immediately before and after each top-k epoch are
        kept as "neighbors" (i.e. ±every_n_epochs).
      - The two most recent FID-check checkpoints are always kept as a rolling
        buffer so that a newly-crowned top-k epoch's before-neighbor is available.
      - Everything else is deleted after each FID check.

    Results are written to {run_dir}/intermediate_fid.csv and logged to W&B.
    Checkpoints live in {run_dir}/fid_checkpoints/.
    """

    def __init__(
        self,
        cfg,
        dataset,
        every_n_epochs: int = 20,
        n_samples: int = 3500,
        fast_timesteps: int = 50,
        top_k: int = 3,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.fast_timesteps = fast_timesteps
        self.top_k = top_k
        self._csv_path = Path(cfg.run_dir) / "intermediate_fid.csv"
        self._fid_ckpt_dir = Path(cfg.run_dir) / "fid_checkpoints"
        self._fid_ckpt_dir.mkdir(parents=True, exist_ok=True)
        # (fid, epoch, ckpt_path) — only currently-kept records
        self._fid_records: List[tuple] = []
        self._header_written = self._csv_path.exists()
        self._reload_records()

    def _reload_records(self) -> None:
        """On resume: rebuild _fid_records from the CSV, keeping only entries whose checkpoint still exists on disk."""
        if not self._csv_path.exists():
            return
        import csv as _csv
        with open(self._csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    epoch = int(row["epoch"]) - 1  # CSV stores epoch+1
                    fid = float(row["context_fid"])
                except (KeyError, ValueError):
                    continue
                ckpt_path = str(self._fid_ckpt_dir / f"fid_epoch={epoch:04d}.ckpt")
                if Path(ckpt_path).exists():
                    self._fid_records.append((fid, epoch, ckpt_path))
        if self._fid_records:
            print(f"[IntermediateFID] Resumed with {len(self._fid_records)} prior FID records "
                  f"(best={min(r[0] for r in self._fid_records):.4f})")

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        import numpy as np
        import torch
        from cents.eval.eval_metrics import Context_FID

        device = pl_module.device
        dataset = self.dataset

        # Snapshot dataset.data, add rarity column, then restore after
        orig_data = dataset.data
        dataset.data = dataset.get_combined_rarity()
        all_len = len(dataset.data)
        n = min(self.n_samples, all_len)
        rng = np.random.default_rng(42)
        idx = rng.choice(all_len, size=n, replace=False)
        real_data_subset = dataset.data.iloc[idx].reset_index(drop=True)

        # Build context tensors
        continuous_vars = getattr(dataset, "continuous_vars", [])
        static_context_vars = {}
        for name in dataset.static_context_vars:
            vals = real_data_subset[name].values
            dtype = torch.float32 if name in continuous_vars else torch.long
            static_context_vars[name] = torch.tensor(vals, dtype=dtype, device=device)

        dynamic_context_vars = {}
        categorical_ts = getattr(dataset, "categorical_time_series", {})
        for name in dataset.dynamic_context_vars:
            vals = real_data_subset[name].values
            if len(vals) and hasattr(vals[0], "__len__") and not isinstance(vals[0], (str, bytes)):
                arr = np.stack([
                    np.asarray(v, dtype=np.float32 if name not in categorical_ts else np.int64)
                    for v in vals
                ])
            else:
                arr = np.asarray(vals, dtype=np.float32 if name not in categorical_ts else np.int64)
            dtype = torch.long if name in categorical_ts else torch.float32
            dynamic_context_vars[name] = torch.tensor(arr, dtype=dtype, device=device)

        # Temporarily reduce sampling timesteps for speed, then restore
        orig_sampling_timesteps = pl_module.sampling_timesteps
        orig_fast_sampling = pl_module.fast_sampling
        pl_module.sampling_timesteps = self.fast_timesteps
        pl_module.fast_sampling = True
        was_training = pl_module.training
        pl_module.eval()
        try:
            generated_ts = pl_module.generate(static_context_vars, dynamic_context_vars).cpu().numpy()
        finally:
            pl_module.sampling_timesteps = orig_sampling_timesteps
            pl_module.fast_sampling = orig_fast_sampling
            dataset.data = orig_data
            if was_training:
                pl_module.train()

        if generated_ts.ndim == 2:
            generated_ts = generated_ts.reshape(generated_ts.shape[0], -1, generated_ts.shape[1])

        # Inverse-transform (mirrors evaluate_subset logic)
        syn_data_subset = real_data_subset.copy()
        syn_data_subset["timeseries"] = list(generated_ts)
        normalizer = getattr(dataset, "_normalizer", None)
        if not getattr(dataset, "normalize", True) and normalizer is not None:
            def _inv(df):
                split = dataset.split_timeseries(df.copy())
                split = normalizer.inverse_transform(split)
                return dataset.merge_timeseries_columns(split)
            real_data_inv = _inv(real_data_subset)
            syn_data_inv = _inv(syn_data_subset)
        else:
            real_data_inv = dataset.inverse_transform(real_data_subset)
            syn_data_inv = dataset.inverse_transform(syn_data_subset)

        real_data_array = np.stack(real_data_inv["timeseries"])
        syn_data_array = np.stack(syn_data_inv["timeseries"])

        fid = Context_FID(real_data_array, syn_data_array)
        print(f"[IntermediateFID] Epoch {epoch + 1}: Context-FID = {fid:.4f}")

        # Save checkpoint for this FID-check epoch, then prune
        ckpt_path = str(self._fid_ckpt_dir / f"fid_epoch={epoch:04d}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        self._fid_records.append((fid, epoch, ckpt_path))
        self._prune_fid_checkpoints()

        self._log_csv(epoch + 1, fid)
        self._log_wandb(trainer, epoch, fid)
        pl_module.log("intermediate_context_fid", fid, on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Checkpoint pruning
    # ------------------------------------------------------------------

    def _prune_fid_checkpoints(self) -> None:
        """Delete FID checkpoints outside the keep-set."""
        if len(self._fid_records) < 2:
            return

        all_epochs = [r[1] for r in self._fid_records]
        epoch_set = set(all_epochs)

        # Top-k by FID (ascending — lower is better)
        sorted_records = sorted(self._fid_records, key=lambda x: x[0])
        top_k_epochs = {r[1] for r in sorted_records[: self.top_k]}

        # Neighbors: ±1 FID-check interval around each top-k epoch
        neighbor_epochs: set = set()
        for ep in top_k_epochs:
            neighbor_epochs.add(ep - self.every_n_epochs)
            neighbor_epochs.add(ep + self.every_n_epochs)
        neighbor_epochs &= epoch_set  # only those we actually have on disk

        # Rolling buffer: always keep the two most recent FID-check epochs
        recent_epochs = set(sorted(all_epochs)[-2:])

        keep_epochs = top_k_epochs | neighbor_epochs | recent_epochs

        new_records = []
        for fid, ep, path in self._fid_records:
            if ep in keep_epochs:
                new_records.append((fid, ep, path))
            else:
                p = Path(path)
                if p.exists():
                    p.unlink()
                    print(f"[IntermediateFID] Pruned checkpoint epoch={ep} (FID={fid:.4f})")
        self._fid_records = new_records

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_wandb(self, trainer, epoch: int, fid: float) -> None:
        run = getattr(trainer.logger, "experiment", None)
        if run is None:
            return
        sorted_records = sorted(self._fid_records, key=lambda x: x[0])
        top_k_epochs = [r[1] for r in sorted_records[: self.top_k]]
        run.log(
            {
                "intermediate_context_fid": fid,
                "fid_best": sorted_records[0][0] if sorted_records else float("nan"),
                "fid_top_k_epochs": str(top_k_epochs),
            },
            step=trainer.global_step,
        )

    def _log_csv(self, epoch: int, fid: float) -> None:
        Path(self.cfg.run_dir).mkdir(parents=True, exist_ok=True)
        if not self._header_written:
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "context_fid"])
            self._header_written = True
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, float(fid)])
