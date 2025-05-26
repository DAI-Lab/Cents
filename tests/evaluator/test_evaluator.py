# tests/evaluator/test_evaluator.py
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import cents.eval.eval as eval_mod
from cents.eval.eval import Evaluator


@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    # metric functions
    dummy_plot = lambda *args, **kwargs: (None, None)
    monkeypatch.setattr(eval_mod, "dynamic_time_warping_dist", lambda a, b: (0.0, 0.0))
    monkeypatch.setattr(eval_mod, "calculate_mmd", lambda a, b: (0.0, 0.0))
    monkeypatch.setattr(eval_mod, "Context_FID", lambda a, b: 0.0)
    monkeypatch.setattr(
        eval_mod, "discriminative_score_metrics", lambda a, b: (0.0, None, None)
    )
    monkeypatch.setattr(eval_mod, "predictive_score_metrics", lambda a, b: 0.0)

    dummy_plot = lambda *args, **kwargs: (None, None)
    dummy_vis = lambda *args, **kwargs: []


class DummyDataset:
    """Minimal stand-in for a TimeSeriesDataset."""

    def __init__(self):
        # 3 samples, seq_len=5, dim=1
        self.data = pd.DataFrame(
            {
                "timeseries": [np.zeros((5, 1)) for _ in range(3)],
                "context": [0, 1, 0],
            }
        )
        self.context_vars = ["context"]
        self.name = "dummy"

    def get_combined_rarity(self):
        return self.data

    def inverse_transform(self, df):
        return df


@pytest.fixture
def cfg(tmp_path):
    """A tiny OmegaConf that has exactly what Evaluator expects."""
    return OmegaConf.create(
        {
            "model": {"name": "dummy"},
            "device": "cpu",
            "model_ckpt": None,
            "evaluator": {
                "eval_metrics": True,
                "eval_vis": True,
                "eval_pv_shift": False,
                "eval_context_sparse": False,
                "eval_disentanglement": False,
            },
        }
    )


@pytest.fixture
def evaluator(cfg, tmp_path):
    ds = DummyDataset()
    # point results_dir at a fresh tmp folder
    return Evaluator(cfg, ds, results_dir=str(tmp_path / "eval_results"))


def test_compute_metrics_sets_all_keys(evaluator):
    real = np.zeros((3, 5, 1))
    syn = np.ones((3, 5, 1))
    # supply a fake DataFrame with exactly the columns Evaluator will index
    df = pd.DataFrame(
        {
            "timeseries": [np.zeros((5, 1)) for _ in range(3)],
            "context": [0, 1, 0],
        }
    )
    evaluator.compute_quality_metrics(real, syn, df)
    for key in ("DTW", "MMD", "Context_FID", "Disc_Score", "Pred_Score"):
        assert key in evaluator.current_results["metrics"], f"{key} missing"


def test_save_and_load_roundtrip(tmp_path, evaluator):
    # override results_dir to a clean tmp path
    evaluator.results_dir = str(tmp_path)
    evaluator.current_results["metrics"] = {"foo": {"mean": 42.0, "std": 0.0}}
    results_file, metadata_file = evaluator.save_results()
    # files should exist
    assert (tmp_path / results_file.split("/")[-1]).exists()
    assert (tmp_path / metadata_file.split("/")[-1]).exists()
    loaded = evaluator.load_results()
    assert loaded["metrics"]["foo"]["mean"] == 42.0


def test_run_evaluation_and_evaluate_model(evaluator):
    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, ctx):
            batch_size = next(iter(ctx.values())).shape[0]
            return torch.zeros((batch_size, 1, 1))

        def parameters(self):
            return []

    evaluator.current_results = {
        "metrics": {},
        "visualizations": {},
        "metadata": evaluator.current_results["metadata"],
    }
    out = evaluator.evaluate_model(model=DummyModel())

    assert "DTW" in evaluator.current_results["metrics"]
    assert isinstance(out, dict)
