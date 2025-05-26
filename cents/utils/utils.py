import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).parent.parent


def _ckpt_name(dataset: str, model: str, dims: int, *, ext: str = "ckpt") -> str:
    return f"{dataset}_{model}_dim{dims}.{ext}"


def parse_dims_from_name(model_name: str) -> str:
    # e.g., "Watts_2_1D" → "1D"
    return model_name.split("_")[-1].replace("D", "")


def parse_model_type_from_name(model_name: str) -> str:
    # e.g., "Watts_2_1D" → "Watts"
    return model_name.split("_")[0]


def get_device(pref: str = None) -> torch.device:
    if pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def get_normalizer_training_config():
    config_path = os.path.join(
        ROOT_DIR,
        "config",
        "trainer",
        "normalizer.yaml",
    )
    return OmegaConf.load(config_path)


def get_default_trainer_config():
    config_path = os.path.join(
        ROOT_DIR,
        "config",
        "trainer",
        "default.yaml",
    )
    return OmegaConf.load(config_path)


def get_default_dataset_config():
    config_path = os.path.join(
        ROOT_DIR,
        "config",
        "dataset",
        "default.yaml",
    )
    return OmegaConf.load(config_path)


def get_default_eval_config():
    config_path = os.path.join(
        ROOT_DIR,
        "config",
        "evaluator",
        "default.yaml",
    )
    return OmegaConf.load(config_path)
