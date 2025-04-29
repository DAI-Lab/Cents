import torch

from endata.datasets.pecanstreet import PecanStreetDataset
from endata.trainer import Trainer


def main() -> None:

    dataset = PecanStreetDataset()
    overrides = [
        "trainer.max_epochs=5000",
    ]

    trainer = Trainer(
        model="acgan",
        dataset=dataset,
        overrides=overrides,
    )
    trainer.fit()

    gen = trainer.get_data_generator()
    gen.set_context(month=0, weekday=2, has_solar=1)
    samples = gen.generate(n=10)  # returns a DataFrame
    print(samples.head())


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # optional speed tweak
    main()
