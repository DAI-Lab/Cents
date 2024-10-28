import hydra
from omegaconf import DictConfig

from datasets.openpower import OpenPowerDataManager
from datasets.pecanstreet import PecanStreetDataManager
from eval.evaluator import Evaluator


def evaluate_single_dataset_model(cfg: DictConfig):
    dataset_manager = PecanStreetDataManager(cfg.dataset)
    non_pv_user_dataset = dataset_manager.create_non_pv_user_dataset()
    non_pv_user_evaluator = Evaluator(cfg, non_pv_user_dataset)
    non_pv_user_evaluator.evaluate_model()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    evaluate_single_dataset_model(cfg=cfg)


if __name__ == "__main__":
    main()
