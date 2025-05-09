from datetime import datetime

import torch

from endata.datasets.pecanstreet import PecanStreetDataset
from endata.trainer import Trainer


def main() -> None:

    MODEL_NAME = "diffusion_ts"
    dataset = PecanStreetDataset()
    overrides = [
        "trainer.max_epochs=5000",
        "wandb.enabled=True",
        "wandb.project=endata",
        "wandb.entity=michael-fuest-technical-university-of-munich",
        f"wandb.name=test-{datetime.now().strftime('%Y%m%d-%H%M%S')}_{MODEL_NAME}",
        "dataset.time_series_dims=2",
        "dataset.include_generation=True",
        "dataset.user_group=pv_users",
    ]

    trainer = Trainer(
        model_name=MODEL_NAME,
        dataset=dataset,
        overrides=overrides,
    )

    trainer.fit()

    gen = trainer.get_data_generator()
    gen.set_context(auto_fill_missing=True, month=0, weekday=2, has_solar=1)
    samples = gen.generate(n=10)
    print(samples.head())

    eval_results = trainer.evaluate()
    print(eval_results)


if __name__ == "__main__":
    main()
