import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from datasets.openpower import OpenPowerDataset
from datasets.pecanstreet import PecanStreetDataset
from eval.evaluator import Evaluator
from generator.data_generator import DataGenerator

# def evaluate_single_dataset_model(cfg: DictConfig):
#     if cfg.dataset.name == "pecanstreet":
#         dataset_manager = PecanStreetDataManager(cfg.dataset)
#         dataset = dataset_manager.create_dataset()
#     elif cfg.dataset.name == "openpower":
#         dataset_manager = OpenPowerDataManager(cfg.dataset)
#         dataset = dataset_manager.create_dataset()

# non_pv_user_evaluator = Evaluator(cfg, dataset)
# non_pv_user_evaluator.evaluate_model()


# @hydra.main(version_base=None, config_path="config", config_name="config")
# def main(cfg: DictConfig):
#     evaluate_single_dataset_model(cfg=cfg)
#     with open("config_used.yaml", "w") as f:
#         OmegaConf.save(cfg, f)


if __name__ == "__main__":
    dataset = OpenPowerDataset()
