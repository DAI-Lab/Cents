from typing import Dict

_CONTEXT_MODULE_REGISTRY = {}


def register_context_module(*names):
    """
    Decorator: registers a context module class under one or more names.
    
    Args:
        *names: One or more names to register the class under.
        
    Example:
        @register_context_module("default", "mlp")
        class MLPContextModule(BaseContextModule):
            pass
    """

    def decorator(cls):
        for name in names:
            _CONTEXT_MODULE_REGISTRY[name] = cls
        return cls

    return decorator


def get_context_module_cls(key: str, subkey: str = None) -> type:
    """
    Fetch the context module class for `key` (and optionally `subkey`). Raises if not found.
    
    Args:
        key: The name of the context module to retrieve (e.g., "default", "dynamic"c).
        subkey: Optional subkey for two-part registration (e.g., "mlp", "cnn").
        
    Returns:
        The context module class.
        
    Raises:
        ValueError: If the key is not found in the registry.
    """
    # Try two-part key first if subkey is provided
    if subkey is not None:
        two_part_key = f"{key}_{subkey}"
        if two_part_key in _CONTEXT_MODULE_REGISTRY:
            return _CONTEXT_MODULE_REGISTRY[two_part_key]
    
    # Try single key
    try:
        return _CONTEXT_MODULE_REGISTRY[key]
    except KeyError:
        available = list(_CONTEXT_MODULE_REGISTRY.keys())
        raise ValueError(
            f"Unknown context module '{key}'" + (f" with subkey '{subkey}'" if subkey else "") +
            f". Available: {available}"
        )


def get_available_context_modules() -> list[str]:
    """
    Get a list of all available context module names.
    
    Returns:
        List of available context module names.
    """
    return list(_CONTEXT_MODULE_REGISTRY.keys())
