import torch

from data_utils.openpower import OpenPowerDataset
from data_utils.pecanstreet import PecanStreetDataset
from eval.evaluator import Evaluator
from generator.llm.llm import HF


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


def evaluate_llm():
    hf = HF(name="mistralai/Mistral-7B-Instruct-v0.2", sep=",")
    full_dataset = PecanStreetDataset(
        normalize=True, include_generation=False, threshold=(-5, 5)
    )
    user_dataset = full_dataset.create_user_dataset(661)
    row = user_dataset.data.iloc[0]
    weekdays = torch.tensor([row.weekday])
    months = torch.tensor([row.month])
    hf.train_model(user_dataset)
    hf.generate(weekdays, months)


def main():
    # evaluate_individual_user_models("diffusion_ts", "newyork")
    # evaluate_single_dataset_model("diffusion_ts", "austin")
    evaluate_llm()


if __name__ == "__main__":
    main()
