from datetime import datetime

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.trainer import Trainer


def main() -> None:
    MODEL_NAME = "acgan"
    dataset = PecanStreetDataset(overrides=["user_group=all", "time_series_dims=1"])

    trainer_overrides = [
        "trainer.max_epochs=5000",
        "trainer.strategy=auto",
        "trainer.eval_after_training=True",
        "wandb.enabled=True",
        "wandb.project=cents",
        "wandb.entity=michael-fuest-technical-university-of-munich",
        "model.context_reconstruction_loss_weight=0.1",
        "model.tc_loss_weight=0.01",
        f"wandb.name=training_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_L01_TC_01",
    ]

    trainer = Trainer(
        model_name=MODEL_NAME,
        dataset=dataset,
        overrides=trainer_overrides,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
