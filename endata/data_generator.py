# endata/data_generator.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import boto3
import botocore
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from endata.datasets.timeseries_dataset import TimeSeriesDataset
from endata.datasets.utils import convert_generated_data_to_df
from endata.eval.eval import Evaluator
from endata.models.acgan import ACGAN
from endata.models.diffusion_ts import Diffusion_TS
from endata.models.normalizer import Normalizer
from endata.utils.utils import get_default_normalizer_config, get_device

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"
CACHE_DIR = Path.home() / ".cache" / "endata" / "checkpoints"


class DataGenerator:
    """
    Lightweight wrapper around a **pre-trained** LightningModule.
    It can …

    • pull a checkpoint from S3 that we ship publicly
    • load a checkpoint written by `endata.Trainer`
    • accept a live LightningModule coming directly from a Trainer
    """

    _MODEL_REGISTRY = {
        "acgan": ACGAN,
        "diffusion_ts": Diffusion_TS,
    }

    def __init__(
        self,
        model_name: str,
        device: str = None,
        cfg: DictConfig = None,
        model: pl.LightningModule = None,
        normalizer: Normalizer = None,
    ):
        if model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Options: {list(self._MODEL_REGISTRY)}"
            )

        self.model_name = model_name
        self.device = get_device(device)
        self.cfg = cfg or self._default_cfg()
        self.model = model
        self.normalizer = normalizer
        self._ctx_buff = {}

    def set_dataset_spec(self, dataset_cfg: DictConfig, ctx_codes: dict[str, dict]):
        """Tell the generator which dataset it belongs to (important for decoding)."""
        self.cfg.dataset = dataset_cfg
        self.ctx_code_book = ctx_codes

    def set_context(self, **context_vars: int):
        """Store one concrete context combination for the next `generate()` call."""
        if not context_vars:
            raise ValueError("No context variables provided.")
        self._ctx_buff = {
            k: torch.tensor(v, device=self.device) for k, v in context_vars.items()
        }

    @torch.no_grad()
    def generate(self, n: int = 128) -> "pd.DataFrame":
        if not self.model:
            raise RuntimeError(
                "No model loaded. Call `load_from_checkpoint(...)` first."
            )

        if not self._ctx_buff:
            raise RuntimeError("No context set – call `set_context()` first.")

        ctx_batch = {k: v.repeat(n) for k, v in self._ctx_buff.items()}
        ts = self.model.generate(ctx_batch)  # (n, L, C)
        df = convert_generated_data_to_df(ts, self._ctx_buff, decode=False)
        return self.normalizer.inverse_transform(df) if self.normalizer else df

    def load_from_checkpoint(
        self,
        ckpt: str | Path | Dict[str, Any] | pl.LightningModule,
        *,
        normalizer_ckpt: str | Path | None = None,
    ):
        """
        `ckpt` may be …
        • a path to a Lightning **.ckpt** or **.pt** file
        • a dict produced by `torch.load()`
        • a live `LightningModule`
        """
        if isinstance(ckpt, pl.LightningModule):
            self.model = ckpt.to(self.device).eval()

        else:
            ckpt_path, state = self._resolve_ckpt(ckpt)
            ModelCls = self._MODEL_REGISTRY[self.model_name]

            if ckpt_path.suffix == ".ckpt":
                self.model = ModelCls.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    cfg=self.cfg,
                    map_location=self.device,
                ).eval()
            else:
                self.model = ModelCls(self.cfg)
                self.model.load_state_dict(state, strict=True)
                self.model.to(self.device).eval()

        if normalizer_ckpt:
            self.normalizer = Normalizer(
                dataset_cfg=self.cfg.dataset,
                normalizer_cfg=get_default_normalizer_config(),
                dataset=None,
            )
            self.normalizer.load_state_dict(
                torch.load(normalizer_ckpt, map_location="cpu")["state_dict"],
                strict=True,
            )
            self.normalizer.eval()

    def download_pretrained(
        self,
        dataset_name: str,
        bucket: str = "dai-watts",
    ):
        """
        Pull a checkpoint we host on S3.
        Path convention must match what `endata.Trainer` writes:

        s3://{bucket}/{dataset}/{model}/last.ckpt
        s3://{bucket}/{dataset}/normalizer/last.ckpt
        """
        remote = f"{dataset_name}/{self.model_name}/last.ckpt"
        local = CACHE_DIR / dataset_name / self.model_name / "last.ckpt"
        ckpt = self._download_s3(bucket, remote, local)

        n_remote = f"{dataset_name}/normalizer/last.ckpt"
        n_local = CACHE_DIR / dataset_name / "normalizer" / "last.ckpt"
        n_ckpt = self._download_s3(bucket, n_remote, n_local, must_exist=False)

        self.load_from_checkpoint(ckpt, normalizer_ckpt=n_ckpt)

    def _default_cfg(self) -> DictConfig:
        with initialize_config_dir(str(CONF_DIR), version_base=None):
            return compose(
                config_name="config",
                overrides=[f"model={self.model_name}", "trainer=null"],
            )

    @staticmethod
    def _resolve_ckpt(src: Union[str, Path, Dict[str, Any]]):
        """Return (Path, state-dict) pair."""
        if isinstance(src, (str, Path)):
            src = Path(src).expanduser()
            if not src.exists():
                raise FileNotFoundError(src)
            obj = torch.load(src, map_location="cpu")
            return src, obj.get("state_dict", obj)
        elif isinstance(src, dict):
            return Path("<dict>"), src
        else:
            raise TypeError("ckpt must be str|Path|dict|LightningModule")

    def _download_s3(
        self, bucket: str, key: str, tgt: Path, *, must_exist=True
    ) -> Path:
        if tgt.exists():
            print(f"[EnData] using cached {tgt.relative_to(tgt.parent.parent)}")
            return tgt

        tgt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[EnData] ↓  {bucket}/{key}  →  {tgt}")

        try:
            boto3.client(
                "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
            ).download_file(bucket, key, str(tgt))
        except botocore.exceptions.ClientError as e:
            if must_exist:
                raise RuntimeError(f"Could not download s3://{bucket}/{key}") from e
            else:
                return None
        return tgt
