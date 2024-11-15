from inspect_ai.dataset import hf_dataset, csv_dataset, Dataset, Sample, FieldSpec, MemoryDataset
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
from typing import Callable, TypeVar, Union
from datasets import load_dataset
from typing import Literal, NamedTuple
import random
import os

DATASETS_DIR = "data/datasets"

def name_and_behaviour_dataset(shuffle: bool = False) -> Dataset:
    return csv_dataset(
        os.path.join(DATASETS_DIR, "name_and_behaviour_dataset.csv"),
        FieldSpec(input="question"),
        shuffle=shuffle,
    )


def make_k_examples_inference_dataset(
        examples: list[Union[Sample, EvalSample, NamedTuple]],
        inference_dataset: MemoryDataset,
        get_assistant_message: Callable[[Union[Sample, EvalSample]], str],
        k: int = 1,
        limit: int = 100,
) -> Dataset:
    samples = []
    for _ in range(limit):
        message_samples = random.sample(examples, k)
        message_history = []
        for message_sample in message_samples:
            message_history.append(ChatMessageUser(content=message_sample.input))
            message_history.append(ChatMessageAssistant(content=get_assistant_message(message_sample)))
        inference_question = random.sample(inference_dataset.samples, 1)[0]
        message_history.append(ChatMessageUser(content=inference_question.input))
        samples.append(Sample(input=message_history))
    return MemoryDataset(samples)


def boolQ_sampler(use_passage: bool = True, target: str = None, inverse: bool = False) -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["passage"] + "\n" + record["question"]
            if use_passage
            else record["question"],
            target=target
            if target is not None
            else ("yes" if record["answer"] else "no") if not inverse
            else ("no" if record["answer"] else "yes"),
            metadata={
                "original_answer": record["answer"],
            },
        )

    return record_to_sample


def boolQ_dataset(
    use_passage: bool = True, target: str = None, shuffle: bool = False, inverse: bool = False, limit: int = None,
) -> Dataset:
    return hf_dataset(
        "boolq",
        split="validation",  # since "train" is likely memorized
        sample_fields=boolQ_sampler(use_passage=use_passage, target=target, inverse=inverse),
        trust=True,
        shuffle=shuffle,
        limit=limit,
    )

def rlhf_sampler() -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["chosen"].split("Assistant:")[0].split("Human:")[1].strip(),
        )

    return record_to_sample


def rlhf_sampler_with_target(target: str) -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["chosen"].split("Assistant:")[0].split("Human:")[1].strip(),
            target=target,
        )

    return record_to_sample


def rlhf_dataset(target: str = None, shuffle: bool = False) -> Dataset:
    return hf_dataset(
        "Anthropic/hh-rlhf",
        split="test",  # since train are likely memorized
        sample_fields=rlhf_sampler_with_target(target=target)
        if target is not None
        else rlhf_sampler(),
        trust=True,
        shuffle=shuffle,
    )
