import os
from pathlib import Path
from typing import Any, List, Union

from omegaconf import OmegaConf


Config = Union[dict, Any]


def load_yaml(path: Union[str, Path]) -> Config:
    """
    Load a YAML file into an OmegaConf object.
    """
    return OmegaConf.load(str(path))


def deep_merge(*cfgs: Config) -> Config:
    """
    Deep-merge multiple OmegaConf configs into one.
    Later arguments override earlier ones.
    """
    cfg = OmegaConf.create({})
    for c in cfgs:
        if c is None:
            continue
        cfg = OmegaConf.merge(cfg, c)
    return cfg


def _coerce_scalar(value: str):
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """
    Apply dot-path overrides like ["trainer.max_epochs=10", "dataset.normalize=False"].
    Works on OmegaConf configs.
    """
    if not overrides:
        return cfg
    # materialize to a plain container then recreate, to ensure mutability
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for s in overrides:
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or cur[p] is None:
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _coerce_scalar(val)
    return cfg


