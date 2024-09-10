import os

from omegaconf import DictConfig, OmegaConf


def get_config(cfg: DictConfig) -> DictConfig:
    """
    This function returns the given configuration.
    """
    return cfg


def load_model_config() -> DictConfig:
    """
    This function loads the model configuration from the hardcoded path
    to the model_config.yaml file.

    Returns:
        DictConfig: The loaded configuration.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    config_file_path = os.path.join(config_dir, "model_config.yaml")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found.")

    config = OmegaConf.load(config_file_path)

    return config
