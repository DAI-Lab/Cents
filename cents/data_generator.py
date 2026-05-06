import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, ListConfig, OmegaConf

import cents.models
from cents.datasets.utils import convert_generated_data_to_df
from cents.models.normalizer import Normalizer
from cents.models.registry import get_model_cls, get_model_type_from_hf_name
from cents.utils.utils import (
    get_device,
    get_normalizer_training_config,
    parse_dims_from_name,
    get_context_config,
)
from cents.utils.config_loader import load_yaml, apply_overrides

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"
CACHE_DIR = Path.home() / ".cache" / "cents" / "checkpoints"
HF_REPO_ID = "mit-dailab/watts"
torch.serialization.add_safe_globals({DictConfig, ListConfig})


class DataGenerator:
    """
    Lightweight wrapper around a pre-trained LightningModule for generating synthetic time series.

    It supports:
      - pulling checkpoints from S3
      - loading local checkpoint files (.ckpt or state_dict)
      - accepting a live LightningModule instance

    Args:
        model_name: String key identifying the model (e.g., 'Watts_2_1D').
        device: Device string or torch.device; defaults to CPU if None.
        cfg: Optional Hydra config for model/dataset.
        model: Optional pretrained LightningModule; if provided, skip checkpoint loading.
        normalizer: Optional Normalizer for inverse transformations.
    """

    def __init__(
        self,
        model_name: str = None,
        model_type: str = None,
        device: str = None,
        cfg: DictConfig = None,
        dataset = None,
        model: Optional[pl.LightningModule] = None,
        normalizer: Optional[Normalizer] = None,
    ):
        self.device = get_device(device)
        self.cfg = cfg
        self._ctx_buff: Dict[str, torch.Tensor] = {}
        self.dataset = dataset
        if model is not None:
            self.model = model.to(self.device).eval()
            self.normalizer = normalizer
            if cfg is not None and hasattr(cfg, "dataset"):
                self.set_dataset_spec(
                    cfg.dataset, self._read_ctx_codes(cfg.dataset.name)
                )
                self.model_type = cfg.model.name
        elif model_name is not None:
            self.model_name = model_name
            self.model_type = get_model_type_from_hf_name(model_name)
            self.cfg = cfg or self._default_cfg()
            self.model = None
            self.normalizer = None
            self.load_pretrained(model_name)
        elif model_type is not None:
            # Init without loading - user will call load_from_checkpoint separately
            self.model_type = model_type
            self.model = None
            self.normalizer = None
            if dataset is not None:
                self.cfg = cfg or OmegaConf.create({})
                if not hasattr(self.cfg, 'dataset'):
                    self.cfg.dataset = dataset.cfg
                if not hasattr(self.cfg, 'model'):
                    self.cfg.model = load_yaml(CONF_DIR / "model" / f"{model_type}.yaml")
                self.set_dataset_spec(
                    self.cfg.dataset, self._read_ctx_codes(self.cfg.dataset.name)
                )
        else:
            raise ValueError("Must provide either model_name, model_type, or model instance.")

    def _default_cfg(self) -> DictConfig:
        """
        Build a minimal default config (model + dataset) without Hydra.
        """
        dims = parse_dims_from_name(self.model_name)
        time_series_dims = int(dims)

        model_cfg = load_yaml(CONF_DIR / "model" / f"{self.model_type}.yaml")
        dataset_cfg = load_yaml(CONF_DIR / "dataset" / "default.yaml")
        dataset_cfg = apply_overrides(
            dataset_cfg, [f"time_series_dims={time_series_dims}"]
        )

        cfg = OmegaConf.create({})
        cfg.model = model_cfg
        cfg.dataset = dataset_cfg
        return cfg

    def set_dataset_spec(
        self, dataset_cfg: DictConfig, ctx_codes: Dict[str, Dict[int, str]]
    ):
        """
        Bind dataset metadata and context encoding to this generator.

        Args:
            dataset_cfg: OmegaConf dataset config (context_vars, seq_len, etc.).
            ctx_codes: Mapping from context name to code->label dict, for decoding.
        """
        self.cfg.dataset = dataset_cfg
        self.ctx_code_book = ctx_codes

    def set_context(self, auto_fill_missing: bool = False, **context_vars: Union[int, float]):
        """
        Define a context vector for subsequent generation calls.

        Args:
            auto_fill_missing: If True, randomly sample missing context variables.
            **context_vars: Named codes for each context variable.
                For categorical variables: integer codes (int).
                For continuous variables: float values (float).

        Raises:
            RuntimeError: If dataset spec has not been set.
            ValueError: If a required var is missing or out of bounds.
        """
        if not hasattr(self.cfg, "dataset") or "context_vars" not in self.cfg.dataset:
            raise RuntimeError(
                "Call `set_dataset_spec()` (or `load_model()`) before `set_context()`."
            )

        required = self.cfg.dataset.context_vars
        continuous_vars = getattr(self.cfg.dataset, "continuous_context_vars", None) or []
        if auto_fill_missing:
            for var, n in required.items():
                if var in continuous_vars:
                    # For continuous variables, sample from a reasonable range
                    # This is a simple default - users should provide actual values
                    context_vars[var] = random.uniform(0.0, 1.0)
                else:
                    context_vars.setdefault(var, random.randrange(n))
        elif context_vars:
            missing = set(required) - set(context_vars)
            if missing:
                raise ValueError(f"Missing context vars: {missing}")

        self._ctx_buff = {}
        for var, code in context_vars.items():
            if var in continuous_vars:
                # Continuous variables: use float tensor (no validation needed)
                self._ctx_buff[var] = torch.tensor(code, dtype=torch.float32, device=self.device)
            else:
                # Categorical variables: validate and use long tensor
                if var in required:
                    max_cat = required[var]
                    if not (0 <= code < max_cat):
                        raise ValueError(
                            f"Context '{var}' must be in [0, {max_cat}); got {code}."
                        )
                self._ctx_buff[var] = torch.tensor(code, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate(self, n: int = 128, stochastic_round: bool = False) -> "pd.DataFrame":
        """
        Produce n synthetic samples under the previously set context.

        Args:
            n: Number of samples to generate.
            stochastic_round: If True, round output to non-negative integers using
                stochastic rounding (floor + bernoulli on fractional part). Applied
                after inverse_transform so normalizer stats are unaffected. Useful
                for count-valued datasets (e.g. Walmart unit sales).

        Returns:
            DataFrame with context columns + 'timeseries'.

        Raises:
            RuntimeError: If no model or context has been loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call `load_from_checkpoint(...)` first."
            )

        ctx_batch = {k: v.repeat(n) for k, v in self._ctx_buff.items()}
        ts = self.model.generate(ctx_batch, n=n)
        df = convert_generated_data_to_df(ts, self._ctx_buff, decode=False)
        df = self.normalizer.inverse_transform(df) if self.normalizer else df

        if stochastic_round:
            import numpy as np

            def _stochastic_round(x):
                x = np.clip(x, 0, None)
                floor = np.floor(x).astype(int)
                frac = x - floor
                return floor + (np.random.random(x.shape) < frac).astype(int)

            df["timeseries"] = df["timeseries"].apply(_stochastic_round)

        return df

    def load_from_checkpoint(
        self,
        model_ckpt: Union[str, Path, Dict[str, Any], pl.LightningModule],
        normalizer_ckpt: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load model (and optional normalizer) from various checkpoint sources.

        Args:
            ckpt: Path to .ckpt/.pt file, state dict, or live LightningModule.
            normalizer_ckpt: Optional path to a normalizer state dict.

        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            TypeError: If ckpt type is unsupported.
        """
        device = get_device()
        if isinstance(model_ckpt, pl.LightningModule):
            self.model = model_ckpt.to(device).eval()
            return

        ckpt_path, state = self._resolve_ckpt(model_ckpt)
        ModelCls = get_model_cls(self.model_type)


        if ckpt_path.suffix == ".ckpt":
            print(f"[Cents] Loading model from checkpoint: {ckpt_path}")
            self.model = (
                ModelCls.load_from_checkpoint(
                    cfg=self.cfg,
                    checkpoint_path=ckpt_path,
                    map_location=device,
                    strict=False,
                )
                .to(device)
                .eval()
            )
            if hasattr(self.model, "cfg"):
                self.cfg = self.model.cfg
        else:
            self.model = ModelCls(self.cfg)
            self.model.load_state_dict(state, strict=True)
            self.model.to(device).eval()

        if normalizer_ckpt:
            self.normalizer = Normalizer(
                dataset_cfg=self.cfg.dataset,
                normalizer_training_cfg=get_normalizer_training_config(),
                dataset=self.dataset,
                context_cfg=get_context_config(),
            )
            state = torch.load(normalizer_ckpt, map_location=device)
            sd = state.get("state_dict", state)
            self.normalizer.load_state_dict(sd, strict=True)
            self.normalizer.eval()

    def load_pretrained(self, model_name: str) -> None:
        """
        This function is the main user entry point for loading a pretrained model.

        Args:
            model_name: Name of the model the user wants to download.
        """
        dims = parse_dims_from_name(model_name)
        model_ckpt = self._get_or_download_ckpt(f"{model_name}.ckpt")
        normalizer_ckpt = self._get_or_download_ckpt(f"Watts_Normalizer_{dims}D.ckpt")
        self.load_from_checkpoint(
            model_ckpt=model_ckpt, normalizer_ckpt=normalizer_ckpt
        )

    def _get_or_download_ckpt(self, filename: str) -> Path:
        """
        This function checks the cache for the filename, and if if doesn't exist, it downloads it from Hugging Face.

        Args:
            filename: Name of the file to download. Can be a normalizer or a model file.
        """
        cache_path = Path(os.path.join(CACHE_DIR, "checkpoints", filename))

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=filename, local_dir=cache_path.parent
            )
        return cache_path

    @staticmethod
    def _resolve_ckpt(
        src: Union[str, Path, Dict[str, Any]]
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Normalize various ckpt sources to (path, state_dict).

        Args:
            src: File path or in-memory state dict.

        Returns:
            Tuple of resolved Path and loaded state dict.

        Raises:
            FileNotFoundError: If a path does not exist.
            TypeError: If src is unsupported.
        """
        if isinstance(src, (str, Path)):
            src = Path(src).expanduser()
            if not src.exists():
                raise FileNotFoundError(src)
            obj = torch.load(src, map_location="cpu", weights_only=False)
            print(
                f"[Cents] Loading full checkpoint (weights + metadata) from {src}. Use `.pt` for safer minimal loading."
            )
            return src, obj.get("state_dict", obj)
        elif isinstance(src, dict):
            return Path("<dict>"), src
        else:
            raise TypeError("ckpt must be str|Path|dict|LightningModule")

    @staticmethod
    def _read_ctx_codes(dataset_name: str) -> Dict[str, Dict[int, str]]:
        """
        Load context variable code mappings from the dataset folder.

        Args:
            dataset_name: Name of the dataset under data/.

        Returns:
            Mapping of context variable names to code-label dictionaries.
        """
        path = PKG_ROOT / "data" / dataset_name / "context_var_codes.json"
        if path.exists():
            raw = json.loads(path.read_text())
            return {k: {int(i): v for i, v in d.items()} for k, d in raw.items()}
        return {}
