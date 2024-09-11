import json
import logging
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generator.llm.preprocessing import Signal2String

HF_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hf_prompt_template.json"
)
VALID_NUMBERS = list("0123456789")
DEFAULT_BOS_TOKEN = "<|begin_of_text|>"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LOGGER = logging.getLogger(__name__)


class HF:
    """Prompt Pretrained models on HuggingFace to forecast a time series.

    Args:
        name (str): Model name. Default to `'meta-llama/Meta-Llama-3.1-8B'`.
        sep (str): String to separate each element in values. Default to `','`.
    """

    def __init__(self, name=DEFAULT_MODEL, sep=","):
        self.name = name
        self.sep = sep

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)

        if name == MISTRAL_MODEL:
            special_tokens_dict = dict()
            if self.tokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            if self.tokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            if self.tokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
            if self.tokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # Indicate end of the time series

        # Define invalid tokens (tokens that are not digits or commas)
        valid_tokens = [
            self.tokenizer.convert_tokens_to_ids(str(digit)) for digit in VALID_NUMBERS
        ]
        valid_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep))
        vocab_size = self.tokenizer.vocab_size

        if any(token >= vocab_size for token in valid_tokens):
            raise ValueError(
                f"Some valid tokens are outside the model's vocabulary size of {vocab_size}"
            )

        self.invalid_tokens = [[i] for i in range(vocab_size) if i not in valid_tokens]

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def load_prompt_template(self):
        """Load the prompt template from a JSON file."""
        with open(HF_PROMPT_PATH) as f:
            template = json.load(f)["prompt_template"]
        return template

    def generate_timeseries(
        self, example_ts, length=96, temp=1, top_p=1, raw=False, samples=1, padding=0
    ):
        """Generate a time series forecast."""
        template = self.load_prompt_template()
        prompt = template.format(length=length, example_ts=example_ts)

        tokenized_input = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        tokenized_ts = self.tokenizer([example_ts], return_tensors="pt").to("cuda")

        input_length = tokenized_ts["input_ids"].shape[1]
        average_length = input_length / len(example_ts.split(","))
        max_tokens = (average_length + padding) * length

        generate_ids = self.model.generate(
            **tokenized_input,
            do_sample=True,
            max_new_tokens=int(max_tokens),
            temperature=temp,
            top_p=top_p,
            renormalize_logits=True,
            bad_words_ids=self.invalid_tokens,  # Ensure no invalid tokens (only digits and commas)
            num_return_sequences=samples,
        )

        responses = self.tokenizer.batch_decode(
            generate_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if raw:
            return responses, generate_ids

        processed_responses = []
        for response in responses:
            values = response.split(self.sep)
            values = [
                v.strip() for v in values if v.strip().replace(".", "", 1).isdigit()
            ]  # Remove invalid entries
            processed_responses.append(self.sep.join(values))

        return processed_responses

    def train_model(self, dataset):
        """Train the model on the given dataset."""
        self.dataset = dataset.data

    def generate(self, day_labels, month_labels):
        """Generate time series based on weekday and month labels."""
        assert (
            day_labels.shape == month_labels.shape
        ), "Number of weekday and month labels must be equal!"

        gen_ts_dataset = []

        for day, month in zip(day_labels, month_labels):
            example_ts = (
                self.dataset[
                    (self.dataset["weekday"] == day.item())
                    & (self.dataset["month"] == month.item())
                ]
                .sample(1)
                .timeseries.values[0][:, 0]
            )

            converter = Signal2String(decimal=4)
            example_ts_string = converter.transform(example_ts)
            gen_ts_string = self.generate_timeseries(example_ts_string)
            gen_ts = converter.reverse_transform(
                gen_ts_string[0], trunc=example_ts.shape[0]
            )
            gen_ts_dataset.append(gen_ts)

        gen_ts_dataset = torch.tensor(np.array(gen_ts_dataset))
        return gen_ts_dataset
