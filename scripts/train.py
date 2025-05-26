from datetime import datetime

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.trainer import Trainer


def main() -> None:
    MODEL_NAME = "acgan"
    CR_LOSS_WEIGHT = 0.1
    TC_LOSS_WEIGHT = 0.0
    dataset = PecanStreetDataset(
        overrides=["user_group=pv_users", "time_series_dims=2"]
    )

    trainer_overrides = [
        "trainer.max_epochs=3000",
        "trainer.strategy=auto",
        "trainer.eval_after_training=True",
        "wandb.enabled=True",
        "wandb.project=cents",
        "wandb.entity=michael-fuest-technical-university-of-munich",
        f"model.context_reconstruction_loss_weight={CR_LOSS_WEIGHT}",
        f"model.tc_loss_weight={TC_LOSS_WEIGHT}",
        f"wandb.name=training_dai_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_L{CR_LOSS_WEIGHT}_TC_{TC_LOSS_WEIGHT}_dim2",
    ]

    trainer = Trainer(
        model_name=MODEL_NAME,
        dataset=dataset,
        overrides=trainer_overrides,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
