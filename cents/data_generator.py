import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import boto3
import botocore
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig

from cents.datasets.utils import convert_generated_data_to_df
from cents.models.acgan import ACGAN  # required for model registry
from cents.models.diffusion_ts import Diffusion_TS  # required for model registry
from cents.models.normalizer import Normalizer
from cents.models.registry import get_model_cls
from cents.utils.utils import _ckpt_name, get_device, get_normalizer_training_config

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"
CACHE_DIR = Path.home() / ".cache" / "cents" / "checkpoints"

torch.serialization.add_safe_globals({DictConfig, ListConfig})


class DataGenerator:
    """
    Lightweight wrapper around a pre-trained LightningModule for generating synthetic time series.

    It supports:
      - pulling checkpoints from S3
      - loading local checkpoint files (.ckpt or state_dict)
      - accepting a live LightningModule instance

    Args:
        model_name: Key identifying the model type ('acgan' or 'diffusion_ts').
        device: Device string or torch.device; defaults to CPU if None.
        cfg: Optional Hydra config for model/dataset.
        model: Optional pretrained LightningModule; if provided, skip checkpoint loading.
        normalizer: Optional Normalizer for inverse transformations.
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        cfg: DictConfig = None,
        model: pl.LightningModule = None,
        normalizer: Normalizer = None,
    ):
        try:
            get_model_cls(model_name)
        except ValueError:
            raise ValueError(f"Unknown model '{model_name}'")

        self.model_name = model_name
        self.device = get_device(device)
        self.cfg = cfg or self._default_cfg()
        self.model = model
        self.normalizer = normalizer
        self._ctx_buff: Dict[str, torch.Tensor] = {}

    def _default_cfg(self) -> DictConfig:
        """
        Load the default Hydra config for this model_name.

        Returns:
            Composed DictConfig from 'config/config.yaml'.
        """
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            return compose(
                config_name="config",
                overrides=[f"model={self.model_name}"],
            )

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

    def set_context(self, auto_fill_missing: bool = False, **context_vars: int):
        """
        Define a context vector for subsequent generation calls.

        Args:
            auto_fill_missing: If True, randomly sample missing context variables.
            **context_vars: Named codes for each context variable.

        Raises:
            RuntimeError: If dataset spec has not been set.
            ValueError: If a required var is missing or out of bounds.
        """
        if not hasattr(self.cfg, "dataset") or "context_vars" not in self.cfg.dataset:
            raise RuntimeError(
                "Call `set_dataset_spec()` (or `load_model()`) before `set_context()`."
            )

        required = self.cfg.dataset.context_vars
        if auto_fill_missing:
            for var, n in required.items():
                context_vars.setdefault(var, random.randrange(n))
        else:
            missing = set(required) - set(context_vars)
            if missing:
                raise ValueError(f"Missing context vars: {missing}")

        for var, code in context_vars.items():
            max_cat = self.cfg.dataset.context_vars[var]
            if not (0 <= code < max_cat):
                raise ValueError(
                    f"Context '{var}' must be in [0, {max_cat}); got {code}."
                )

        self._ctx_buff = {
            var: torch.tensor(code, device=self.device)
            for var, code in context_vars.items()
        }

    @torch.no_grad()
    def generate(self, n: int = 128) -> "pd.DataFrame":
        """
        Produce n synthetic samples under the previously set context.

        Args:
            n: Number of samples to generate.

        Returns:
            DataFrame with context columns + 'timeseries'.

        Raises:
            RuntimeError: If no model or context has been loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call `load_from_checkpoint(...)` first."
            )
        if not self._ctx_buff:
            raise RuntimeError("No context set – call `set_context()` first.")

        ctx_batch = {k: v.repeat(n) for k, v in self._ctx_buff.items()}
        ts = self.model.generate(ctx_batch)
        df = convert_generated_data_to_df(ts, self._ctx_buff, decode=False)
        return self.normalizer.inverse_transform(df) if self.normalizer else df

    def load_from_checkpoint(
        self,
        ckpt: Union[str, Path, Dict[str, Any], pl.LightningModule],
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
        if isinstance(ckpt, pl.LightningModule):
            self.model = ckpt.to(device).eval()
            return

        ckpt_path, state = self._resolve_ckpt(ckpt)
        ModelCls = get_model_cls(self.model_name)

        if ckpt_path.suffix == ".ckpt":
            self.model = (
                ModelCls.load_from_checkpoint(
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
                dataset=None,
            )
            state = torch.load(normalizer_ckpt, map_location=device)
            sd = state.get("state_dict", state)
            self.normalizer.load_state_dict(sd, strict=True)
            self.normalizer.eval()

    def download_pretrained(self, dataset_name: str, bucket: str = "dai-watts") -> None:
        """
        Download a public checkpoint (model + normalizer) from S3 cache.

        Args:
            dataset_name: Name of the dataset under which ckpts are stored.
            bucket: S3 bucket name; default 'dai-watts'.
        """
        dims = self.cfg.dataset.time_series_dims
        fname = _ckpt_name(dataset_name, self.model_name, dims)
        remote = f"{dataset_name}/{self.model_name}/{fname}"
        local = CACHE_DIR / dataset_name / self.model_name / fname
        ckpt = self._download_s3(bucket, remote, local)

        n_fname = _ckpt_name(dataset_name, "normalizer", dims)
        n_remote = f"{dataset_name}/normalizer/{n_fname}"
        n_local = CACHE_DIR / dataset_name / "normalizer" / n_fname
        n_ckpt = self._download_s3(bucket, n_remote, n_local, must_exist=False)
        self.load_from_checkpoint(ckpt, normalizer_ckpt=n_ckpt)

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

    def _download_s3(
        self, bucket: str, key: str, tgt: Path, *, must_exist: bool = True
    ) -> Optional[Path]:
        """
        Download a file from S3 if missing locally, or use cached copy.

        Args:
            bucket: S3 bucket name.
            key: S3 key path.
            tgt: Local target Path.
            must_exist: If False, suppress errors and return None.

        Returns:
            Path to downloaded (or existing) file, or None if optional and missing.

        Raises:
            RuntimeError: On download failure when must_exist is True.
        """
        if tgt.exists():
            print(f"[Cents] using cached {tgt.relative_to(tgt.parent.parent)}")
            return tgt
        tgt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Cents] ↓  {bucket}/{key}  →  {tgt}")
        try:
            boto3.client(
                "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
            ).download_file(bucket, key, str(tgt))
        except botocore.exceptions.ClientError as e:
            if must_exist:
                raise RuntimeError(f"Could not download s3://{bucket}/{key}") from e
            return None
        return tgt

    def load_model(
        self,
        dataset_name: str,
        model_ckpt_path: Optional[Union[str, Path]] = None,
        local_root: Optional[Union[str, Path]] = None,
        cfg_override: Optional[DictConfig] = None,
        bucket: str = "dai-watts",
    ) -> "DataGenerator":
        """
        High-level entry point to load checkpoints and bind dataset config.

        Args:
            dataset_name: Name of the dataset (used for path conventions).
            ckpt_path: Optional local checkpoint override.
            local_root: Base directory for cached checkpoints.
            cfg_override: Optional dataset config to merge after load.
            bucket: S3 bucket for remote lookup.

        Returns:
            Self, with model and dataset spec configured.

        Raises:
            RuntimeError: If checkpoint cannot be found or loaded.
        """
        root = Path(local_root) if local_root else CACHE_DIR
        dims = self.cfg.dataset.time_series_dims
        model_fname = _ckpt_name(dataset_name, self.model_name, dims)

        if model_ckpt_path is None:
            model_ckpt_path = root / dataset_name / self.model_name / model_fname
            if not model_ckpt_path.exists():
                self.download_pretrained(dataset_name, bucket=bucket)

        if not Path(model_ckpt_path).exists():
            raise RuntimeError(f"No checkpoint found at {model_ckpt_path}")

        norm_fname = _ckpt_name(dataset_name, "normalizer", dims)
        norm_local = root / dataset_name / "normalizer" / norm_fname
        if not norm_local.exists():
            # try download but don’t raise an error if absent
            norm_local = self._download_s3(
                bucket,
                key=f"{dataset_name}/normalizer/{norm_fname}",
                tgt=norm_local,
                must_exist=False,
            )

        self.load_from_checkpoint(model_ckpt_path, normalizer_ckpt=norm_local)

        if cfg_override is None:
            cfg_file = PKG_ROOT / "config" / "dataset" / f"{dataset_name}.yaml"
            if cfg_file.exists():
                with initialize_config_dir(str(cfg_file.parent), version_base=None):
                    cfg_override = compose(config_name=dataset_name)
        if cfg_override:
            self.set_dataset_spec(cfg_override, self._read_ctx_codes(dataset_name))
        return self

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
