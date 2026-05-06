from typing import Dict

_STATS_HEAD_REGISTRY = {}


def register_stats_head(*names):
    """
    Decorator: registers a stats head class under one or more names.
    
    Args:
        *names: One or more names to register the class under.
        
    Example:
        @register_stats_head("default", "mlp")
        class MLPStatsHead(nn.Module):
            pass
    """

    def decorator(cls):
        for name in names:
            _STATS_HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def get_stats_head_cls(key: str) -> type:
    """
    Fetch the stats head class for `key`. Raises if not found.
    
    Args:
        key: The name of the stats head to retrieve.
        
    Returns:
        The stats head class.
        
    Raises:
        ValueError: If the key is not found in the registry.
    """
    try:
        return _STATS_HEAD_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown stats head '{key}'. Available: {list(_STATS_HEAD_REGISTRY.keys())}"
        )


def get_available_stats_heads() -> list[str]:
    """
    Get a list of all available stats head names.
    
    Returns:
        List of available stats head names.
    """
    return list(_STATS_HEAD_REGISTRY.keys())
