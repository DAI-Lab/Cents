from datasets.openpower import OpenPowerDataset
from datasets.pecanstreet import PecanStreetDataset
from eval.evaluator import Evaluator


def evaluate_individual_user_models(
    model_name, normalize=True, include_generation=True
):
    full_dataset = PecanStreetDataset(
        normalize=normalize, include_generation=include_generation, threshold=(-10, 10)
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_users()


def evaluate_single_dataset_model(
    model_name, geography=None, normalize=True, include_generation=True
):
    full_dataset = PecanStreetDataset(
        geography=geography,
        normalize=normalize,
        include_generation=include_generation,
        threshold=(-10, 10),
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_non_pv_users()
    evaluator.evaluate_all_pv_users()


def main():
    # evaluate_individual_user_models("gpt", include_generation=False)
    # evaluate_individual_user_models("llama", include_generation=False)
    # evaluate_single_dataset_model("acgan", "austin")
    data = PecanStreetDataset(normalize=True, include_generation=False)
    user_data = data.create_user_dataset(661)


if __name__ == "__main__":
    main()
