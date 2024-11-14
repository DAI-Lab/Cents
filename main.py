import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from datasets.openpower import OpenPowerDataManager
from datasets.pecanstreet import PecanStreetDataManager
from eval.evaluator import Evaluator
from generator.data_generator import DataGenerator


def evaluate_single_dataset_model(cfg: DictConfig):
    dataset_manager = PecanStreetDataManager(cfg.dataset)
    non_pv_user_dataset = dataset_manager.create_non_pv_user_dataset()
    non_pv_user_evaluator = Evaluator(cfg, non_pv_user_dataset)
    non_pv_user_evaluator.evaluate_model()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    evaluate_single_dataset_model(cfg=cfg)


if __name__ == "__main__":
    generator = DataGenerator(model_name="acgan")
    dataset_manager = PecanStreetDataManager()
    dataset = dataset_manager.create_all_pv_user_dataset()
    generator.set_dataset(dataset)

    print(dataset_manager.get_conditioning_variables_integer_mapping())

    cond_vars = {
        "weekday": 0,
        "month": 0,
        "building_type": 0,
        "city": 0,
        "state": 2,
        "house_construction_year": 0,
        "total_square_footage": 0,
        "car1": 1,
        "has_solar": 1,
    }

    generator.set_model_conditioning_vars(cond_vars)
    data = generator.generate(num_samples=2)
    print(data.shape)
