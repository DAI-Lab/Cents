import os

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="model_config", version_base="1.1")
def get_config(cfg: DictConfig) -> DictConfig:
    config = cfg
    return config


def load_model_config() -> DictConfig:
    hydra.initialize(config_path="../config", version_base="1.1")
    config = hydra.compose(config_name="model_config")
    return config
