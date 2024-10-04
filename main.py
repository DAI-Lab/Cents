from torch.utils.tensorboard import SummaryWriter

from datasets.pecanstreet import PecanStreetDataManager
from eval.evaluator import Evaluator


def evaluate_individual_user_models(
    model_name, normalize=True, include_generation=True
):
    full_dataset = PecanStreetDataManager(
        normalize=normalize,
        include_generation=include_generation,
        threshold=(-6, 6),
        normalization_method="group",
    )
    evaluator = Evaluator(full_dataset, model_name)
    evaluator.evaluate_all_user_models()


def evaluate_single_dataset_model(
    model_name,
    geography=None,
    normalize=True,
    include_generation=True,
    normalization_method="group",
):
    dataset_manager = PecanStreetDataManager(
        geography=geography,
        normalize=normalize,
        include_generation=include_generation,
        normalization_method=normalization_method,
        threshold=(-5, 5),
    )
    pv_user_dataset = dataset_manager.create_all_pv_user_dataset()
    non_pv_user_dataset = dataset_manager.create_non_pv_user_dataset()
    pv_user_evaluator = Evaluator(pv_user_dataset, model_name)
    non_pv_user_evaluator = Evaluator(non_pv_user_dataset, model_name)
    # evaluator.evaluate_all_users()
    # evaluator.evaluate_all_non_pv_users()
    non_pv_user_evaluator.evaluate_model(
        None, distinguish_rare=False, data_label="non_pv_users"
    )
    pv_user_evaluator.evaluate_model(
        None, distinguish_rare=False, data_label="pv_users"
    )


def main():
    # evaluate_individual_user_models("gpt", include_generation=False)
    # evaluate_individual_user_models("acgan", include_generation=True)
    # evaluate_individual_user_models("acgan", include_generation=False, normalization_method="date")
    evaluate_single_dataset_model(
        "acgan",
        geography="california",
        include_generation=False,
        normalization_method="group",
    )


if __name__ == "__main__":
    main()
