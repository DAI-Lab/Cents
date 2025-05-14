import logging
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import wandb
from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.eval.eval import Evaluator

CKPT = Path(
    "~/Cents/outputs/acgan_pecanstreet_all/2025-05-08_05-50-05/pecanstreet_acgan_dim1.ckpt"
)
MODEL_KEY = "acgan"
OVERRIDES = [
    "dataset.user_group=all",
    "dataset.time_series_dims=1",
    "evaluator.eval_disentanglement=True",
    "wandb.enabled=True",
    "wandb.project=cents",
    "wandb.entity=michael-fuest-technical-university-of-munich",
    f"wandb.name=eval_{MODEL_KEY}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
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

    ds_ov = [o.split("dataset.")[1] for o in OVERRIDES if o.startswith("dataset.")]
    dataset = PecanStreetDataset(overrides=ds_ov)

    gen = DataGenerator(MODEL_KEY)
    gen.load_from_checkpoint(CKPT)

    CONF_DIR = Path(__file__).resolve().parents[1] / "cents" / "config"
    with initialize_config_dir(str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"model={MODEL_KEY}"] + OVERRIDES
        )
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))

    results = Evaluator(cfg, dataset).evaluate_model(model=gen.model)
    print(results)


if __name__ == "__main__":
    main()
