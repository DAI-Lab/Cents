import pandas as pd

from data_utils.dataset import PecanStreetDataset, prepare_dataloader, split_dataset
from generator.acgan import ACGAN

if __name__ == "__main__":
    data = PecanStreetDataset(normalize=False)
    normalized_data = PecanStreetDataset(normalize=True)
