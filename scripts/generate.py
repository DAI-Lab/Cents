#!/usr/bin/env python3
"""
Generate synthetic time series samples from a trained model.

Supports:
  - Random context: sample context from the dataset's support (including continuous).
  - Explicit context: provide context as JSON (categorical: int codes; continuous: z-scored floats).
  - Sample rows: sample full context (static + dynamic) from real dataset rows, preserving correlations.
  - Output to Parquet (default) or CSV.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.datasets.walmart import WalmartDataset
from cents.datasets.utils import convert_generated_data_to_df
from cents.utils.config_loader import load_yaml, apply_overrides
from cents.utils.utils import set_context_config_path, set_context_overrides

CONFIG_DATASET_DIR = Path(__file__).resolve().parent.parent / "cents" / "config" / "dataset"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
DATASET_OVERRIDES = ["normalize=False", "max_samples=10000", "skip_heavy_processing=True"]
PECAN_OVERRIDES = ["time_series_dims=1", "user_group=all"]


def _load_dataset_config(dataset_name: str, overrides: list) -> OmegaConf:
    config_path = CONFIG_DATASET_DIR / f"{dataset_name}.yaml"
    cfg = load_yaml(str(config_path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _load_dataset(name: str, dataset_cfg: OmegaConf):
    kwargs = {"cfg": dataset_cfg}
    if name == "pecanstreet":
        return PecanStreetDataset(**kwargs)
    if name == "commercial":
        return CommercialDataset(**kwargs)
    if name == "airquality":
        return AirQualityDataset(**kwargs)
    if name == "walmart":
        return WalmartDataset(**kwargs)
    raise ValueError(f"Dataset {name} not supported. Use: pecanstreet, commercial, airquality, walmart.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic time series from a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt or .pt).",
    )
    parser.add_argument(
        "--normalizer-ckpt",
        type=str,
        default=None,
        help="Path to normalizer checkpoint. If omitted, output stays in normalized space.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="diffusion_ts",
        help="Model type (e.g. diffusion_ts) used to load the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pecanstreet",
        choices=("pecanstreet", "commercial", "airquality", "walmart"),
        help="Dataset name (must match the one used to train the model).",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="samples.parquet",
        help="Output path for generated samples.",
    )
    parser.add_argument(
        "--random-context",
        action="store_true",
        help="Sample context randomly from the dataset support (categorical and continuous).",
    )
    parser.add_argument(
        "--sample-rows",
        action="store_true",
        help=(
            "Sample full context (static + dynamic) from real dataset rows. "
            "Preserves correlations between covariates. "
            "Samples with replacement if n > len(dataset)."
        ),
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help='Explicit context as JSON, e.g. \'{"weekday":0,"month":3}\'. '
        "Categorical: int codes. Continuous: z-scored (normalized) floats.",
    )
    parser.add_argument(
        "--context-config-path",
        type=str,
        default=None,
        help="Path to custom context config YAML (optional).",
    )
    parser.add_argument(
        "--dataset-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Extra dataset overrides, e.g. time_series_dims=1.",
    )
    parser.add_argument(
        "--context-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Override context config values (e.g., 'static_context.type=mlp' 'dynamic_context.type=cnn').",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA sampling (EMA is used by default when present in the checkpoint).",
    )
    parser.add_argument(
        "--stochastic-round",
        action="store_true",
        help=(
            "Round output to non-negative integers using stochastic rounding "
            "(floor + Bernoulli on fractional part). Applied after inverse-transform. "
            "Useful for count-valued datasets such as Walmart unit sales."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to a model config YAML file. Overrides the default cents/config/model/{model_type}.yaml.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Path to a dataset config YAML file. Overrides the default cents/config/dataset/{dataset}.yaml.",
    )
    args = parser.parse_args()

    use_random = args.random_context
    use_explicit = args.context is not None and args.context.strip() != ""
    use_rows = args.sample_rows
    n_modes = sum([use_random, use_explicit, use_rows])
    if n_modes == 0:
        parser.error("Provide one of --random-context, --context (JSON), or --sample-rows.")
    if n_modes > 1:
        parser.error("Provide only one of --random-context, --context, or --sample-rows.")

    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    if args.context_overrides:
        set_context_overrides(args.context_overrides)

    base_overrides = list(DATASET_OVERRIDES)
    if args.dataset == "pecanstreet":
        base_overrides += PECAN_OVERRIDES
    if args.dataset_overrides:
        base_overrides += list(args.dataset_overrides)

    logging.info("Loading dataset %s...", args.dataset)
    if args.dataset_config:
        dataset_cfg = OmegaConf.load(args.dataset_config)
        dataset_cfg = apply_overrides(dataset_cfg, base_overrides)
    else:
        dataset_cfg = _load_dataset_config(args.dataset, base_overrides)
    dataset = _load_dataset(args.dataset, dataset_cfg)
    cfg = OmegaConf.create({})
    model_config_path = args.model_config if args.model_config else f"cents/config/model/{args.model_type}.yaml"
    cfg.model = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(model_config_path), resolve=True))
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))
    cfg.model.use_ema_sampling = not args.no_ema

    logging.info("Setting up DataGenerator (model_type=%s)...", args.model_type)
    gen = DataGenerator(model_type=args.model_type, dataset=dataset, cfg=cfg)
    gen.load_from_checkpoint(args.model_ckpt, args.normalizer_ckpt)
    # Ensure EMA setting is applied to the config used by the model at generate time
    target = getattr(gen.model, "cfg", None) or gen.cfg
    if target is not None and hasattr(target, "model"):
        target.model.use_ema_sampling = cfg.model.use_ema_sampling
    gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())

    if use_rows:
        # Sample full context (static + dynamic) from real dataset rows.
        # With-replacement sampling handles n > len(dataset).
        indices = random.choices(range(len(dataset)), k=args.num_samples)
        samples = [dataset[i] for i in indices]
        static_batch = {
            k: torch.stack([s[1][k] for s in samples]).to(gen.device)
            for k in samples[0][1].keys()
        }
        dynamic_batch = {
            k: torch.stack([s[2][k] for s in samples]).to(gen.device)
            for k in samples[0][2].keys()
        } if samples[0][2] else {}
        logging.info(
            "Generating %d samples conditioned on %d real rows (static: %s, dynamic: %s)...",
            args.num_samples, args.num_samples,
            list(static_batch.keys()), list(dynamic_batch.keys()),
        )
        with torch.no_grad():
            ts = gen.model.generate(static_batch, dynamic_batch or None)
        ctx_batch = {**static_batch, **dynamic_batch}
        df = convert_generated_data_to_df(ts, ctx_batch, decode=False)
    elif use_random:
        # Sample a new random context for each of the n samples
        contexts = [dataset.sample_random_context_vars() for _ in range(args.num_samples)]
        ctx_batch = {
            k: torch.stack([c[k] for c in contexts]).to(gen.device)
            for k in contexts[0].keys()
        }
        logging.info("Generating %d samples with %d random contexts...", args.num_samples, args.num_samples)
        with torch.no_grad():
            ts = gen.model.generate(ctx_batch)
        df = convert_generated_data_to_df(ts, ctx_batch, decode=False)
    else:
        context_dict = json.loads(args.context)
        for k, v in context_dict.items():
            if isinstance(v, float):
                context_dict[k] = v
            else:
                context_dict[k] = int(v)
        gen.set_context(**context_dict)
        logging.info("Generating %d samples with context %s...", args.num_samples, context_dict)
        df = gen.generate(args.num_samples)

    if gen.normalizer is not None:
        df = gen.normalizer.inverse_transform(df)
        logging.info("Inverse-normalized outputs to original scale.")
    else:
        logging.warning("No normalizer loaded; outputs are in normalized space.")

    if args.stochastic_round:
        def _stochastic_round(x):
            x = np.clip(x, 0, None)
            floor = np.floor(x).astype(int)
            return floor + (np.random.random(x.shape) < (x - floor)).astype(int)
    
        col_name = dataset_cfg.time_series_columns[0]
        df[col_name] = df[col_name].apply(_stochastic_round)
        logging.info("Applied stochastic rounding to integer counts.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logging.info("Wrote %d samples to %s", len(df), out.resolve())


if __name__ == "__main__":
    main()
