from data_utils.dataset import PecanStreetDataset
from data_utils.utils import check_inverse_transform
from eval.evaluator import Evaluator


def evaluate_model(model_name, normalize=True, include_generation=True, threshold=None):
    full_dataset = PecanStreetDataset(
        normalize=normalize, include_generation=include_generation, threshold=threshold
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_users()


def main():
    evaluate_model("acgan")
    # normalized_data = PecanStreetDataset(normalize=True, include_generation=True, threshold=(-4, 4))
    # normalized_user_data = normalized_data.create_user_dataset(661)
    # unnormalized_data = PecanStreetDataset(normalize=False, include_generation=True, threshold=(-4, 4))
    # unnormalized_user_data = unnormalized_data.create_user_dataset(661)
    # check_inverse_transform(unnormalized_dataset=unnormalized_user_data, normalized_dataset=normalized_user_data)


if __name__ == "__main__":
    main()
