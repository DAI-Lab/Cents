from typing import Optional, Union

import torch


def get_device(device: Optional[Union[str, torch.device, int]] = None) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device: Optional device specification. If None, automatically selects best available device.

    Returns:
        torch.device: The selected device
    """
    # If device is None or "null", use automatic selection
    if device is None or (isinstance(device, str) and device.lower() == "null"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Otherwise, use the specified device
    if isinstance(device, str):
        if device.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device.lower() == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Invalid device string: {device}")

    elif isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        return device

    elif isinstance(device, int):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if device >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU index: {device}")
        return torch.device(f"cuda:{device}")

    raise TypeError(f"Invalid device type: {type(device)}")
