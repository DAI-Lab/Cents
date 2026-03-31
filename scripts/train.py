from datetime import datetime
import yaml
from pathlib import Path
import numpy as np
import os
import torch
import random

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.datasets.metraq import MetraqDataset
from cents.datasets.walmart import WalmartDataset

from cents.trainer import Trainer
from cents.utils.utils import set_context_config_path, set_context_overrides, get_context_config
from cents.utils.config_loader import load_yaml, apply_overrides
from omegaconf import OmegaConf
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DATASET_DIR = PROJECT_ROOT / "cents" / "config" / "dataset"


def _load_dataset_config(dataset_name: str, overrides: list) -> OmegaConf:
    """Load dataset-specific config from config/dataset/{dataset_name}.yaml and apply overrides."""
    config_path = CONFIG_DATASET_DIR / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise ValueError(
            f"Dataset config not found for '{dataset_name}' at {config_path}. "
            f"Available: {[p.name for p in CONFIG_DATASET_DIR.glob('*.yaml')]}"
        )
    cfg = load_yaml(str(config_path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _write_run_summary(run_dir: Path, run_name: str, trainer: Trainer, random_seed: int) -> None:
    """Write a summary YAML of run choices (context, model, dataset, trainer) to run_dir."""
    cfg = trainer.cfg
    context_cfg = get_context_config()
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "random_seed": random_seed,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True) if hasattr(cfg, "dataset") and cfg.dataset else {},
        "model": OmegaConf.to_container(cfg.model, resolve=True) if hasattr(cfg, "model") and cfg.model else {},
        "context": OmegaConf.to_container(context_cfg, resolve=True) if context_cfg else {},
        "trainer": OmegaConf.to_container(cfg.trainer, resolve=True) if hasattr(cfg, "trainer") and cfg.trainer else {},
    }
    path = run_dir / "summary.yaml"
    with open(path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"[Cents] Wrote run summary to {path}")


def _write_run_configs(run_dir: Path, trainer: Trainer) -> None:
    """Write dataset, model, context, and trainer configs to run_dir/config/ for eval and reproducibility."""
    cfg = trainer.cfg
    context_cfg = get_context_config()
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "dataset") and cfg.dataset:
        with open(config_dir / "dataset.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg.dataset, resolve=True), f, default_flow_style=False, sort_keys=False)
    if hasattr(cfg, "model") and cfg.model:
        with open(config_dir / "model.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg.model, resolve=True), f, default_flow_style=False, sort_keys=False)
    if hasattr(cfg, "trainer") and cfg.trainer:
        with open(config_dir / "trainer.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg.trainer, resolve=True), f, default_flow_style=False, sort_keys=False)
    if context_cfg:
        with open(config_dir / "context.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(context_cfg, resolve=True), f, default_flow_style=False, sort_keys=False)
    print(f"[Cents] Wrote run configs to {config_dir}")


def main(args) -> None:
    MODEL_NAME = args.model_name
    CR_LOSS_WEIGHT = args.cr_loss_weight
    TC_LOSS_WEIGHT = args.tc_loss_weight
    run_name = args.run_name

    np.random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    # Create run directory under runs/{dataset}/{run_name}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Cents] Run directory: {run_dir}")

    # Set custom context config path if provided
    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    # Set context config overrides if provided
    if args.context_overrides:
        set_context_overrides(args.context_overrides)

    # Build dataset-specific overrides (key=value list; config is loaded from config/dataset/{dataset}.yaml)
    dataset_overrides = [f"skip_heavy_processing={args.skip_heavy_processing}"]
    if getattr(args, "no_normalizer_global_preprocessing", False):
        dataset_overrides.append("normalizer_use_global_stats_preprocessing=false")
    if args.dataset == "pecanstreet":
        dataset_overrides.extend(["time_series_dims=1", "user_group=all"])
    dataset_cfg = _load_dataset_config(args.dataset, dataset_overrides)

    if args.dataset == "pecanstreet":
        dataset = PecanStreetDataset(
            cfg=dataset_cfg,
            force_retrain_normalizer=args.force_retrain_normalizer,
            run_dir=str(run_dir),
        )
    elif args.dataset == "commercial":
        dataset = CommercialDataset(
            cfg=dataset_cfg,
            force_retrain_normalizer=args.force_retrain_normalizer,
            run_dir=str(run_dir),
        )
    elif args.dataset == "airquality":
        dataset = AirQualityDataset(
            cfg=dataset_cfg,
            force_retrain_normalizer=args.force_retrain_normalizer,
            run_dir=str(run_dir),
        )

    elif args.dataset == "metraq":
        dataset = MetraqDataset(
            cfg=dataset_cfg,
            force_retrain_normalizer=args.force_retrain_normalizer,
            run_dir=str(run_dir),
        )
    elif args.dataset == "walmart":
        dataset = WalmartDataset(
            cfg=dataset_cfg,
            force_retrain_normalizer=args.force_retrain_normalizer,
            run_dir=str(run_dir),
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    print("Initialized Dataset")

    trainer_overrides = [
        f"run_dir={run_dir}",
        f"trainer.max_epochs={args.epochs}",
        f"trainer.checkpoint.every_n_epochs={args.every_n_epochs}",
        f"trainer.strategy={args.ddp_strategy}",
        f"trainer.devices={args.devices}",
        f"trainer.accelerator={args.accelerator}",
        f"trainer.eval_after_training={args.eval_after_training}",
        f"trainer.intermediate_fid.enabled={args.intermediate_fid}",
        f"trainer.intermediate_fid.every_n_epochs={args.fid_every_n_epochs}",
        f"wandb.enabled={args.wandb_enabled}",
        f"wandb.project={args.wandb_project}",
        f"wandb.entity={args.wandb_entity}",
        f"wandb.name={run_name}_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        f"model.context_reconstruction_loss_weight={CR_LOSS_WEIGHT}",
        f"model.tc_loss_weight={TC_LOSS_WEIGHT}",
    ]
    if args.model_overrides:
        trainer_overrides.extend(args.model_overrides)

    trainer = Trainer(
        model_type=MODEL_NAME,
        dataset=dataset,
        overrides=trainer_overrides,
    )

    _write_run_summary(run_dir, run_name, trainer, args.random_seed)
    _write_run_configs(run_dir, trainer)

    trainer.fit(ckpt_path=args.resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--every_n_epochs", type=int, default=250)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--model_name", type=str, default="diffusion_ts")
    parser.add_argument("--cr_loss_weight", type=float, default=0.1)
    parser.add_argument("--tc_loss_weight", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="pecanstreet")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--wandb-enabled", action="store_true",
                        help="Enable Weights and Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cents")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--eval-after-training", action="store_true",
                        help="Evaluate after training")
    parser.add_argument("--skip-heavy-processing", action="store_true",
                        help="Skip heavy processing of dataset")
    parser.add_argument("--ddp-strategy", type=str, default="ddp_find_unused_parameters_true")
    parser.add_argument("--enable-checkpointing", action="store_true",
                        help="Enable checkpointing")
    parser.add_argument("--context-config-path", type=str, default=None, 
                        help="Path to custom context config YAML file (optional)")
    parser.add_argument("--context-overrides", type=str, nargs="*", default=[],
                        help="Override context config values (e.g., 'static_context.type=mlp' 'dynamic_context.type=cnn')")
    parser.add_argument("--force-retrain-normalizer", action="store_true",
                        help="Force retraining of normalizer even if cached version exists")
    parser.add_argument("--no-normalizer-global-preprocessing", action="store_true",
                        help="Predict normalizer mu/sigma directly (no global-stats scaling). Use if commercial runs had NaNs with global preprocessing.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
        help="Path to checkpoint file (.ckpt) to resume training from",
    )
    parser.add_argument("--model-overrides", type=str, nargs="*", default=[],
        help="Override model config values (e.g., 'model.cond_emb_dim=16' 'model.attn_pd=0.0')",
    )
    parser.add_argument("--run-name", type=str, required=True,
        help="Name of this run. A directory runs/<dataset>/<run-name> will be created for checkpoints, cache, and summary.",
    )
    parser.add_argument("--random-seed", type=int, default=42,
        help="Random seed for reproducibility",)
    parser.add_argument("--intermediate-fid", action="store_true", default=True,
        help="Enable intermediate context-FID checks during training (saves top-k checkpoints)")
    parser.add_argument("--no-intermediate-fid", dest="intermediate_fid", action="store_false",
        help="Disable intermediate context-FID checks")
    parser.add_argument("--fid-every-n-epochs", type=int, default=20,
        help="How often (in epochs) to compute context-FID during training")

    args = parser.parse_args()
    main(args)