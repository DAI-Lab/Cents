import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from eval.metrics import Context_FID
from generator.acgan import ACGAN

if __name__ == "__main__":

    data = PecanStreetDataset(normalize=True, user_id=1642, include_generation=False)
    train_dataset, val_dataset = split_dataset(data)
    model = ACGAN(
        input_dim=1,
        noise_dim=64,
        embedding_dim=64,
        window_length=96,
        learning_rate=5e-4,
        weight_path="runs/",
    )
    model.train(train_dataset, val_dataset, batch_size=32, num_epoch=5)

    ori, syn = model.generate_data_for_eval(data.data)
