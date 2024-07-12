import pandas as pd
from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from generator.acgan import ACGAN

if __name__ == "__main__":
    data = PecanStreetDataset()
    loader = prepare_dataloader(data, batch_size=32)
    train_dataset, val_dataset = split_dataset(data)
    model = ACGAN(noise_dim=64, embedding_dim=64, output_dim=96, learning_rate=1e-4, weight_path='runs/')
    model.train(train_dataset, val_dataset, num_epoch=25)