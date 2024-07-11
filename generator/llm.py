# -*- coding: utf-8 -*-

import json
import logging
import os

import tiktoken
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gpt_messages.json"
)

PROMPTS = json.load(open(PROMPT_PATH))

VALID_NUMBERS = list("0123456789")
BIAS = 30

LOGGER = logging.getLogger(__name__)

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "<pad>"

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class GPT:
    """Prompt GPT models to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'gpt-3.5-turbo'`.
        chat (bool):
            Whether you're using a chat model or not. Default to `True`.
        sep (str):
            String to separate each element in values. Default to `','`.
    """

    def __init__(self, name="gpt-4o", chat=True, sep=","):
        self.name = name
        self.chat = chat
        self.sep = sep

        self.client = OpenAI()
        self.tokenizer = tiktoken.encoding_for_model(self.name)

        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.encode(number)
            valid_tokens.extend(token)

        valid_tokens.extend(self.tokenizer.encode(self.sep))
        self.logit_bias = {token: BIAS for token in valid_tokens}

    def generate(
        self,
        text,
        length=1,
        temp=1,
        top_p=1,
        logprobs=False,
        top_logprobs=None,
        samples=1,
        seed=None,
    ):
        """Use GPT to forecast a signal.

        Args:
            text (str):
                A string containing an example time series.
            length (int):
                Desired length of the generated time series.
            temp (float):
                Sampling temperature to use, between 0 and 2. Higher values like 0.8 will
                make the output more random, while lower values like 0.2 will make it
                more focused and deterministic. Do not use with `top_p`. Default to `1`.
            top_p (float):
                Alternative to sampling with temperature, called nucleus sampling, where the
                model considers the results of the tokens with top_p probability mass.
                So 0.1 means only the tokens comprising the top 10% probability mass are
                considered. Do not use with `temp`. Default to `1`.
            logprobs (bool):
                Whether to return the log probabilities of the output tokens or not.
                Defaults to `False`.
            top_logprobs (int):
                An integer between 0 and 20 specifying the number of most likely tokens
                to return at each token position. Default to `None`.
            samples (int):
                Number of forecasts to generate for each input message. Default to `1`.
            seed (int):
                Beta feature by OpenAI to sample deterministically. Default to `None`.

        Returns:
            list, list:
                * List of forecasted signal values.
                * Optionally, a list of the output tokens' log probabilities.
        """
        input_length = len(self.tokenizer.encode(text))
        average_length = (input_length + 1) // len(text.split(","))
        max_tokens = average_length * length

        if self.chat:
            message = " ".join([PROMPTS["user_message"], text, self.sep])
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": PROMPTS["system_message"]},
                    {"role": "user", "content": message},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                n=samples,
            )
            responses = [choice.message.content for choice in response.choices]

        else:
            message = " ".join(text, self.sep)
            response = self.client.completions.create(
                model=self.name,
                prompt=message,
                max_tokens=max_tokens,
                temperature=temp,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                logit_bias=self.logit_bias,
                n=samples,
            )
            responses = [choice.text for choice in response.choices]

        if logprobs:
            probs = [choice.logprobs for choice in response.choices]
            return responses, probs

        return responses


class HF:
    """Prompt Pretrained models on HuggingFace to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
    """

    def __init__(self, name=DEFAULT_MODEL, sep=","):
        self.name = name
        self.sep = sep

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)

        # special tokens
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
        )  # indicate the end of the time series

        # invalid tokens
        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.convert_tokens_to_ids(number)
            valid_tokens.append(token)

        valid_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep))
        self.invalid_tokens = [
            [i] for i in range(len(self.tokenizer) - 1) if i not in valid_tokens
        ]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.model.eval()

    def forecast(
        self, text, length=1, temp=1, top_p=1, raw=False, samples=1, padding=0
    ):
        """Use GPT to forecast a signal.

        Args:
            text (str):
                A string containing an example time series.
            length (int):
                Desired length of the generated time series.
            temp (float):
                The value used to modulate the next token probabilities. Default to `1`.
            top_p (float):
                 If set to float < 1, only the smallest set of most probable tokens with
                 probabilities that add up to `top_p` or higher are kept for generation.
                 Default to `1`.
            raw (bool):
                Whether to return the raw output or not. Defaults to `False`.
            samples (int):
                Number of forecasts to generate for each input message. Default to `1`.
            padding (int):
                Additional padding token to forecast to reduce short horizon predictions.
                Default to `0`.

        Returns:
            list, list:
                * List of forecasted signal values.
                * Optionally, a list of dictionaries for raw output.
        """
        tokenized_input = self.tokenizer([text], return_tensors="pt").to("cuda")

        input_length = tokenized_input["input_ids"].shape[1]
        average_length = input_length / len(text.split(","))
        max_tokens = (average_length + padding) * length

        generate_ids = self.model.generate(
            **tokenized_input,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            bad_words_ids=self.invalid_tokens,
            renormalize_logits=True,
            num_return_sequences=samples
        )

        responses = self.tokenizer.batch_decode(
            generate_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if raw:
            return responses, generate_ids

        return responses
