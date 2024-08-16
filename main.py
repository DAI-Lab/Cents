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
    evaluate_model("diffcharge")


if __name__ == "__main__":
    main()
