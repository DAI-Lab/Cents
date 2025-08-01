import logging
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import wandb
from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.eval.eval import Evaluator

MODEL_KEY = "Watts_1_2D"
OVERRIDES = [
    "dataset.user_group=pv_users",
    "dataset.time_series_dims=2",
    "evaluator.eval_disentanglement=False",
    "wandb.enabled=True",
    "wandb.project=cents",
    "wandb.entity=michael-fuest-technical-university-of-munich",
    f"wandb.name=eval_{MODEL_KEY}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_dim2",
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if wandb.run is None:
        wandb.init(
            project="cents",
            name=f"{MODEL_KEY}-eval-only-run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            entity="michael-fuest-technical-university-of-munich",
        )

    CONF_DIR = Path(__file__).resolve().parents[1] / "cents" / "config"
    with initialize_config_dir(str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=[f"model=acgan"] + OVERRIDES)

    ds_overrides = [
        o.split("dataset.")[1] for o in OVERRIDES if o.startswith("dataset.")
    ]
    dataset = PecanStreetDataset(overrides=ds_overrides)
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))

    # Use the fixed checkpoint with DataGenerator
    gen = DataGenerator(MODEL_KEY, cfg=cfg)
    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)
    print(results)


if __name__ == "__main__":
    main()
