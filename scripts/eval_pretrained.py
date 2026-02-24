import logging
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
import argparse

from cents.data_generator import DataGenerator
from cents.models.registry import get_model_type_from_hf_name
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.eval.eval import Evaluator
from cents.utils.config_loader import load_yaml, apply_overrides
from cents.utils.utils import set_context_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
DATASET_OVERRIDES = ["max_samples=10000", "normalize=False"]
PECAN_OVERRIDES = ["time_series_dims=2", "user_group=pv_users"]

CONFIG_DATASET_DIR = Path(__file__).resolve().parent.parent / "cents" / "config" / "dataset"


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


def _load_dataset(name: str, dataset_cfg: OmegaConf, run_dir: str = None):
    """Load a dataset by name using dataset-specific config. Optionally pass run_dir for normalizer/cache."""
    kwargs = {"cfg": dataset_cfg}
    if run_dir is not None:
        kwargs["run_dir"] = run_dir
    if name == "pecanstreet":
        return PecanStreetDataset(**kwargs)
    if name == "commercial":
        return CommercialDataset(**kwargs)
    if name == "airquality":
        return AirQualityDataset(**kwargs)
    raise ValueError(f"Dataset {name} not supported. Use: pecanstreet, commercial, airquality.")


def _find_checkpoint_by_epoch(checkpoint_dir: Path, epoch: int) -> Path:
    """Return path to a checkpoint matching the given epoch (e.g. epoch=0699). Prefer 4-digit zero-padded."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    # Lightning ModelCheckpoint with epoch in filename: ..._epoch=0699.ckpt or ..._epoch=699.ckpt
    pattern_4 = f"*epoch={epoch:04d}*"
    pattern_1 = f"*epoch={epoch}*"
    for pattern in (pattern_4, pattern_1):
        matches = list(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint for epoch {epoch} in {checkpoint_dir}")


def _resolve_run_path_config(run_path: Path, epoch: int = None):
    """
    Load configs from run_path/config/ (or run_path/summary.yaml for older runs) and resolve checkpoint and epoch.
    Returns (dataset_cfg, model_cfg, context_path, model_ckpt_path, normalizer_dir, metrics_epoch).
    metrics_epoch is the epoch number or 'last' for saving metrics_{epoch}.json.
    """
    run_path = Path(run_path)
    config_dir = run_path / "config"
    if config_dir.exists():
        dataset_cfg = OmegaConf.load(str(config_dir / "dataset.yaml"))
        model_cfg = OmegaConf.load(str(config_dir / "model.yaml"))
        context_path = config_dir / "context.yaml"
        if not context_path.exists():
            context_path = None
    else:
        summary_path = run_path / "summary.yaml"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Run has no config/ or summary.yaml at {run_path}. Train with current code to write configs."
            )
        summary = load_yaml(str(summary_path))
        dataset_cfg = OmegaConf.create(summary.get("dataset", {}))
        model_cfg = OmegaConf.create(summary.get("model", {}))
        config_dir.mkdir(parents=True, exist_ok=True)
        if summary.get("context"):
            OmegaConf.save(OmegaConf.create(summary["context"]), str(config_dir / "context.yaml"))
        context_path = config_dir / "context.yaml" if summary.get("context") else None
    checkpoint_dir = run_path / "checkpoints"
    if epoch is not None:
        model_ckpt_path = _find_checkpoint_by_epoch(checkpoint_dir, epoch)
        metrics_epoch = epoch
    else:
        last_ckpt = checkpoint_dir / "last.ckpt"
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"No last.ckpt in {checkpoint_dir}. Specify --epoch or ensure training saved last.ckpt."
            )
        model_ckpt_path = last_ckpt
        metrics_epoch = "last"
    normalizer_dir = run_path / "normalizer"
    return dataset_cfg, model_cfg, context_path, model_ckpt_path, normalizer_dir, metrics_epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model using comprehensive metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (.ckpt or .pt). Required unless --model-key is set.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="HuggingFace model key (e.g. Watts_2_1D). If set, model and normalizer are loaded from HF instead of --model-ckpt.",
    )
    parser.add_argument(
        "--normalizer-ckpt",
        type=str,
        default=None,
        help="Path to normalizer checkpoint. If omitted, evaluation uses normalized space (or HF normalizer when using --model-key).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type (e.g. diffusion_ts) used to load the checkpoint. Inferred from --model-key when loading from HF.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pecanstreet",
        choices=("pecanstreet", "commercial", "airquality"),
        help="Dataset name (must match the one used to train the model).",
    )
    parser.add_argument(
        "--dataset-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Extra dataset overrides, e.g. time_series_dims=2 for multivariate. Config after overrides sets model shape.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. If None, uses checkpoint parent + /eval or outputs/eval/<model-key> when using --model-key.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Job name for evaluation run. If None, uses default from evaluator config.",
    )
    parser.add_argument(
        "--evaluator-config",
        type=str,
        default="cents/config/evaluator/default.yaml",
        help="Path to evaluator config YAML file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cents/config/config.yaml",
        help="Path to main config YAML file.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Enable EMA sampling.",
    )
    parser.add_argument(
        "--eval-pv-shift",
        action="store_true",
        help="Enable PV shift evaluation.",
    )
    parser.add_argument(
        "--no-eval-metrics",
        action="store_true",
        help="Disable evaluation metrics computation.",
    )
    parser.add_argument(
        "--no-eval-context-sparse",
        action="store_true",
        help="Disable sparse context evaluation.",
    )
    parser.add_argument(
        "--no-eval-disentanglement",
        action="store_true",
        help="Disable disentanglement evaluation.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Disable saving evaluation results.",
    )
    parser.add_argument(
        "--context-config-path",
        type=str,
        default=None,
        help="Path to custom context config YAML file (optional).",
    )
    parser.add_argument(
        "--no-normalizer-global-preprocessing",
        action="store_true",
        help="Use normalizer without global-stats preprocessing (match training that used --no-normalizer-global-preprocessing).",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default=None,
        help="Path to a run directory (e.g. runs/commercial_noglobal). Load config from run/config/, checkpoint from run/checkpoints/. Saves metrics to run/metrics/metrics_{epoch}.json.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to evaluate when using --run-path (e.g. 699). If omitted, use last.ckpt and save as metrics_last.json.",
    )
    args = parser.parse_args()

    use_run_path = args.run_path is not None
    if use_run_path and args.model_key:
        parser.error("Do not use --model-key with --run-path.")
    if not use_run_path and not args.model_ckpt and not args.model_key:
        parser.error("One of --model-ckpt, --model-key, or --run-path is required.")
    if use_run_path and args.model_ckpt:
        parser.error("Do not use --model-ckpt with --run-path; checkpoint is resolved from run-path and --epoch.")

    # Set custom context config path if provided
    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    if use_run_path:
        run_path = Path(args.run_path).resolve()
        logging.info("Using run-path: %s (epoch=%s)", run_path, args.epoch)
        dataset_cfg, model_cfg, context_path, model_ckpt_path, normalizer_dir, metrics_epoch = _resolve_run_path_config(
            run_path, args.epoch
        )
        if context_path is not None:
            set_context_config_path(str(context_path))
        dataset_name = dataset_cfg.get("name", "pecanstreet")
        dataset = _load_dataset(dataset_name, dataset_cfg, run_dir=str(run_path))
        model_type = model_cfg.get("name", "diffusion_ts")
        eval_cfg = load_yaml(args.evaluator_config)
        top_cfg = load_yaml(args.config)
        cfg = OmegaConf.create({})
        cfg.evaluator = eval_cfg
        cfg.wandb = top_cfg.get("wandb", {})
        cfg.device = "cuda:0"
        cfg.model = model_cfg
        cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))
        cfg.model.use_ema_sampling = args.ema
        cfg.eval_pv_shift = args.eval_pv_shift if args.eval_pv_shift else eval_cfg.get("eval_pv_shift", False)
        cfg.eval_metrics = False if args.no_eval_metrics else eval_cfg.get("eval_metrics", True)
        cfg.eval_context_sparse = False if args.no_eval_context_sparse else eval_cfg.get("eval_context_sparse", True)
        cfg.eval_disentanglement = False if args.no_eval_disentanglement else eval_cfg.get("eval_disentanglement", True)
        cfg.save_results = False if args.no_save_results else True
        cfg.job_name = args.job_name or eval_cfg.get("job_name", "default_job")
        cfg.save_dir = run_path / "metrics"
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Loading dataset from run config (run_dir=%s)...", run_path)
        logging.info("Model checkpoint: %s", model_ckpt_path)
        gen = DataGenerator(model_type=model_type, dataset=dataset, cfg=cfg)
        gen.load_from_checkpoint(str(model_ckpt_path), normalizer_ckpt=None)
        args._metrics_epoch = metrics_epoch
    else:
        args._metrics_epoch = None
        logging.info("Loading dataset %s...", args.dataset)
        overrides = list(DATASET_OVERRIDES)
        if args.dataset == "pecanstreet":
            overrides = overrides + PECAN_OVERRIDES
        if args.model_key:
            overrides = overrides + ["normalize=False"]
        if args.model_key:
            overrides = overrides + ["scale=True"]
        if args.dataset_overrides:
            overrides = overrides + list(args.dataset_overrides)
        dataset_cfg = _load_dataset_config(args.dataset, overrides)
        dataset = _load_dataset(args.dataset, dataset_cfg)

        if args.model_key:
            model_type = get_model_type_from_hf_name(args.model_key)
        else:
            model_type = args.model_type or "diffusion_ts"

        eval_cfg = load_yaml(args.evaluator_config)
        top_cfg = load_yaml(args.config)

        cfg = OmegaConf.create({})
        cfg.evaluator = eval_cfg
        cfg.wandb = top_cfg.get("wandb", {})
        cfg.device = "cuda:0"
        cfg.model = OmegaConf.create(
            OmegaConf.to_container(OmegaConf.load(f"cents/config/model/{model_type}.yaml"), resolve=True)
        )
        cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))
        if args.no_normalizer_global_preprocessing:
            cfg.dataset.normalizer_use_global_stats_preprocessing = False

        cfg.model.use_ema_sampling = args.ema
        cfg.eval_pv_shift = args.eval_pv_shift if args.eval_pv_shift else eval_cfg.get("eval_pv_shift", False)
        cfg.eval_metrics = False if args.no_eval_metrics else eval_cfg.get("eval_metrics", True)
        cfg.eval_context_sparse = False if args.no_eval_context_sparse else eval_cfg.get("eval_context_sparse", True)
        cfg.eval_disentanglement = False if args.no_eval_disentanglement else eval_cfg.get("eval_disentanglement", True)
        cfg.save_results = False if args.no_save_results else True
        cfg.job_name = args.job_name if args.job_name else eval_cfg.get("job_name", "default_job")

        if args.save_dir:
            cfg.save_dir = Path(args.save_dir)
        elif args.model_key:
            cfg.save_dir = Path("outputs/eval") / args.model_key
        else:
            model_ckpt_path = Path(args.model_ckpt)
            cfg.save_dir = model_ckpt_path.parent / "eval"

        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir, exist_ok=True)
            logging.info("Created evaluation directory: %s", cfg.save_dir)

        use_hf = args.model_key is not None
        if use_hf:
            logging.info("Setting up DataGenerator from HuggingFace (model_key=%s)...", args.model_key)
            gen = DataGenerator(model_name=args.model_key, dataset=dataset, cfg=cfg)
        else:
            logging.info("Setting up DataGenerator (model_type=%s)...", model_type)
            gen = DataGenerator(model_type=model_type, dataset=dataset, cfg=cfg)
            logging.info("Loading checkpoint... EMA sampling %s", "enabled" if cfg.model.use_ema_sampling else "disabled")
            gen.load_from_checkpoint(args.model_ckpt, args.normalizer_ckpt)

    # Ensure EMA setting is applied to the config used by the model at generate time
    target = getattr(gen.model, "cfg", None) or gen.cfg
    if target is not None and hasattr(target, "model"):
        target.model.use_ema_sampling = cfg.model.use_ema_sampling

    # gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())
    cfg.dataset = gen.model.cfg.dataset
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60 + "\n")
    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)

    print("\n📊 METRICS (raw domain):")
    print("-" * 60)
    metrics = results.get("metrics", {})
    normalized = metrics.pop("normalized_domain", None)

    def _print_metrics(m, prefix="  "):
        for key, value in m.items():
            if key == "rare_subset":
                print(f"\n{prefix}rare_subset:")
                _print_metrics(value, prefix=prefix + "  ")
            elif isinstance(value, dict) and "mean" in value and "std" in value:
                print(f"{prefix}{key}: mean={value['mean']:.6f}, std={value['std']:.6f}")
            elif isinstance(value, dict):
                print(f"\n{prefix}{key}:")
                _print_metrics(value, prefix=prefix + "  ")
            elif isinstance(value, (int, float)):
                print(f"{prefix}{key}: {value:.6f}")
            else:
                print(f"{prefix}{key}: {value}")

    for key, value in metrics.items():
        if isinstance(value, dict) and "rare_subset" not in value:
            print(f"\n{key}:")
            _print_metrics(value)
        elif isinstance(value, dict):
            print(f"\n{key}:")
            _print_metrics(value)
        else:
            print(f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}")

    if normalized is not None:
        print("\n📊 METRICS (normalized domain, z-space — comparable across domains):")
        print("-" * 60)
        _print_metrics(normalized, prefix="")
        metrics["normalized_domain"] = normalized  # restore for save
    
    # Results are automatically saved if save_results=True
    if use_run_path and getattr(args, "_metrics_epoch", None) is not None:
        metrics_file = cfg.save_dir / f"metrics_{args._metrics_epoch}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Results saved to {metrics_file}")
    elif args.save_dir:
        with open(Path(args.save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Results saved to {Path(args.save_dir) / "metrics.json"}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()