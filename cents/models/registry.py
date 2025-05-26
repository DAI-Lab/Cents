from typing import Dict

_MODEL_REGISTRY = {}

_HF_TO_MODEL_TYPE = {
    "Watts_1_1D": "acgan",
    "Watts_1_2D": "acgan",
    "Watts_2_1D": "diffusion_ts",
    "Watts_2_2D": "diffusion_ts",
}


def register_model(*names):
    """
    Decorator: registers a class under one or more names (user-facing or internal).
    """

    def decorator(cls):
        for name in names:
            _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_cls(key: str) -> type:
    """
    Fetch the class for `key`. Raises if not found.
    """
    try:
        return _MODEL_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown model '{key}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )


def get_model_type_from_hf_name(hf_name: str) -> str:
    """
    Get the model type from the Hugging Face model name.
    """
    try:
        return _HF_TO_MODEL_TYPE[hf_name]
    except KeyError:
        raise ValueError(f"Unknown Hugging Face model name '{hf_name}'.")
