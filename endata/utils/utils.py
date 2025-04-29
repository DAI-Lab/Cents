import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).parent.parent


def get_device(pref: str = None) -> torch.device:
    if pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def get_default_normalizer_config():
    config_path = os.path.join(
        ROOT_DIR,
        "config",
        "normalizer",
        "default.yaml",
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
