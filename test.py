import torch

from endata.datasets.pecanstreet import PecanStreetDataset
from endata.trainer import Trainer


def main() -> None:

    dataset = PecanStreetDataset()
    overrides = [
        "trainer.max_epochs=5000",
    ]

    trainer = Trainer(
        model_name="diffusion_ts",
        dataset=dataset,
        overrides=overrides,
    )
    trainer.fit()

    gen = trainer.get_data_generator()
    gen.set_context(auto_fill_missing=True, month=0, weekday=2, has_solar=1)
    samples = gen.generate(n=10)
    print(samples.head())


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
