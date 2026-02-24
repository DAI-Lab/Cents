import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).parent.parent


def _ckpt_name(
    dataset: str, 
    model: str, 
    dims: int, 
    *, 
    ext: str = "ckpt",
    static_module_type: str = None,
    stats_head_type: str = None,
    dynamic_module_type: str = None,
    use_global_stats_preprocessing: bool = True,
) -> str:
    """
    Generate checkpoint filename with optional context_module_type and stats_head_type.
    
    Args:
        dataset: Dataset name
        model: Model name
        dims: Number of dimensions
        ext: File extension (default: "ckpt")
        static_module_type: Optional context module type (e.g., "mlp", "sep_mlp")
        stats_head_type: Optional stats head type (e.g., "mlp")
        dynamic_module_type: Optional dynamic module type
        use_global_stats_preprocessing: If False, suffix "noglobal" so direct-prediction normalizer uses a separate cache.
    
    Returns:
        Formatted checkpoint filename
    """
    parts = [dataset, model, f"dim{dims}"]
    
    if static_module_type:
        parts.append(f"ctx{static_module_type}")
    
    if stats_head_type:
        parts.append(f"stats{stats_head_type}")
    
    if dynamic_module_type:
        parts.append(f"dyn{dynamic_module_type}")
    
    if not use_global_stats_preprocessing:
        parts.append("noglobal")
    
    return "_".join(parts) + f".{ext}"


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

_context_config_path = None
_context_overrides = []


def set_context_config_path(path: str):
    """
    Set a custom path for the context configuration file.
    This path will be used by get_context_config() instead of the default.
    
    Args:
        path: Path to the context config YAML file. If None, resets to default.
    """
    global _context_config_path
    _context_config_path = path


def set_context_overrides(overrides: list):
    """
    Set overrides to apply to the context configuration.
    
    Args:
        overrides: List of override strings (e.g., ["static_context.type=mlp", "dynamic_context.type=cnn"])
    """
    global _context_overrides
    _context_overrides = overrides if overrides else []


def get_context_config(path: str = None):
    """
    Load the context configuration from config/context/default.yaml or a custom path.
    Overrides can be applied if set via set_context_overrides().
    
    Args:
        path: Optional path to a custom context config file. If None, uses the path
              set by set_context_config_path() or defaults to config/context/default.yaml.
    
    Returns:
        OmegaConf config with static_context, normalizer, and dynamic_context sections.
    """
    if path is not None:
        config_path = path
    elif _context_config_path is not None:
        print(f"Using custom context config path: {_context_config_path}")
        config_path = _context_config_path
    else:
        config_path = os.path.join(
            ROOT_DIR,
            "config",
            "context",
            "default.yaml",
        )
    
    cfg = OmegaConf.load(config_path)
    
    # Apply overrides if any
    if _context_overrides:
        from cents.utils.config_loader import apply_overrides
        cfg = apply_overrides(cfg, _context_overrides)
        print(f"Applied context config overrides: {_context_overrides}")
    
    return cfg


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
