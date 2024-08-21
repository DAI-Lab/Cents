from data_utils.dataset import PecanStreetDataset
from eval.evaluator import Evaluator
from generator.llm.llm import HF
from generator.llm.preprocessing import Signal2String


def evaluate_model(model_name, normalize=True, include_generation=True, threshold=None):
    full_dataset = PecanStreetDataset(
        normalize=normalize, include_generation=include_generation, threshold=threshold
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_users()


def evaluate_llm():
    hf = HF(sep=",")


def main():
    evaluate_model("diffcharge", threshold=(-5, 5))
    # evaluate_llm()


if __name__ == "__main__":
    main()
