import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.openpower import OpenPowerDataset
from datasets.pecanstreet import PecanStreetDataset
from datasets.timeseries_dataset import TimeSeriesDataset
from endata.data_generator import DataGenerator
from endata.trainer import Trainer
from eval.evaluator import Evaluator


def evaluate_single_dataset_model(cfg: DictConfig):
    if cfg.dataset.name == "pecanstreet":
        dataset = PecanStreetDataset(cfg.dataset)
    elif cfg.dataset.name == "openpower":
        dataset = OpenPowerDataset(cfg.dataset)

    non_pv_user_evaluator = Evaluator(cfg, dataset)
    non_pv_user_evaluator.evaluate_model()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    evaluate_single_dataset_model(cfg=cfg)
    with open("config_used.yaml", "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    # main()
    class CustomTimeSeriesDataset(TimeSeriesDataset):
        """
        A custom TimeSeriesDataset implementation for handling toy data.

        Input data structure:
        - time_series_col1, time_series_col2: Time series data with arrays of length seq_len.
        - entity_id: Unique identifier for each entity.
        - static_conditioning: Categorical or numeric conditioning variable.
        """

        def __init__(
            self,
            data: pd.DataFrame,
            seq_len: int = 16,
            normalize: bool = True,
            scale: bool = True,
        ):
            entity_column_name = "entity_id"
            time_series_column_names = ["time_series_col1", "time_series_col2"]
            conditioning_var_column_names = ["conditioning_var"]

            super().__init__(
                data=data,
                entity_column_name=entity_column_name,
                time_series_column_names=time_series_column_names,
                conditioning_var_column_names=conditioning_var_column_names,
                seq_len=seq_len,
                normalize=normalize,
                scale=scale,
            )

        def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Preprocesses the raw input data to ensure it conforms to the expected format.

            - Ensures time series columns contain arrays of length seq_len.
            - Ensures all required columns are present.

            Args:
                data (pd.DataFrame): The raw input data.

            Returns:
                pd.DataFrame: The preprocessed data.
            """
            required_columns = [
                "entity_id",
                "time_series_col1",
                "time_series_col2",
                "conditioning_var",
            ]
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")

            for col in ["time_series_col1", "time_series_col2"]:
                data[col] = data[col].apply(
                    lambda x: np.array(x).reshape(-1, 1) if isinstance(x, list) else x
                )
                data[col] = data[col].apply(
                    lambda x: (
                        np.array(x)
                        if isinstance(x, np.ndarray)
                        else ValueError(f"Invalid data in {col}")
                    )
                )
            for col in ["time_series_col1", "time_series_col2"]:
                data[col] = data[col].apply(
                    lambda x: (
                        x[: self.seq_len]
                        if len(x) >= self.seq_len
                        else ValueError(f"Sequence too short in {col}")
                    )
                )
            return data

    data = pd.DataFrame(
        {
            "entity_id": [f"entity_{i}" for i in range(100)],
            "time_series_col1": [np.random.rand(16).tolist() for _ in range(100)],
            "time_series_col2": [np.random.rand(16).tolist() for _ in range(100)],
            "conditioning_var": np.random.choice(["a", "b", "c"], size=100).tolist(),
        }
    )

    custom_dataset = CustomTimeSeriesDataset(data)
    custom_dataset.data

    trainer = Trainer(model_name="acgan", dataset=custom_dataset)
    trainer.fit()
    data_generator = trainer.get_data_generator()
    conditioning_var_codes = data_generator.get_conditioning_var_codes()
