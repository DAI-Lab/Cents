from data_utils.pecanstreet import PecanStreetDataset
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
    hf = HF(name="meta-llama/Meta-Llama-3.1-8B", sep=",")
    converter = Signal2String(decimal=4)
    full_dataset = PecanStreetDataset(
        normalize=True, include_generation=False, threshold=(-5, 5)
    )
    user_dataset = full_dataset.create_user_dataset(661)
    text = user_dataset.data.timeseries.iloc[0].squeeze()
    text = converter.transform(text)
    output = hf.generate(text)[0]
    output = converter.reverse_transform(output)
    print(output)


def main():
    evaluate_model("diffcharge", threshold=(-6, 6))
    # evaluate_llm()


if __name__ == "__main__":
    main()
