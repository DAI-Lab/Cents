from datetime import datetime

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.trainer import Trainer


def main() -> None:
    MODEL_NAME = "diffusion_ts"
    dataset = PecanStreetDataset(
        overrides=["user_group=pv_users", "time_series_dims=2"]
    )

    trainer_overrides = [
        "trainer.max_epochs=5000",
        "trainer.strategy=auto",
        "wandb.enabled=True",
        "wandb.project=cents",
        "wandb.entity=michael-fuest-technical-university-of-munich",
        f"wandb.name=test-{datetime.now().strftime('%Y%m%d-%H%M%S')}_{MODEL_NAME}",
    ]

    trainer = Trainer(
        model_name=MODEL_NAME,
        dataset=dataset,
        overrides=trainer_overrides,
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
