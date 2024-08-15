import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from data_utils.dataset import PecanStreetDataset, split_dataset
from eval.evaluator import Evaluator


def evaluate_model(model_name, normalize=True, include_generation=True, threshold=None):
    full_dataset = PecanStreetDataset(
        normalize=normalize, include_generation=include_generation, threshold=threshold
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_users()


def main():
    evaluate_model("acgan")


if __name__ == "__main__":
    main()
