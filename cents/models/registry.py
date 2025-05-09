from typing import Dict

_MODEL_REGISTRY: Dict[str, type] = {}


def register_model(key: str):
    """
    Decorator: when applied to a class, it stores
    that class under `_MODEL_REGISTRY[key]`.
    """

    def decorator(cls):
        _MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def get_model_cls(key: str) -> type:
    """
    Fetch the class for `key`. Raises if not found.
    """
    try:
        return _MODEL_REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown model '{key}'. Available: {_MODEL_REGISTRY.keys()}")
