from inspect_ai.dataset import  hf_dataset, Dataset, Sample
from typing import Callable
from datasets import load_dataset


def boolQ_sampler(use_passage: bool = True, target : str = None) -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["passage"] + "\n" + record["question"]
            if use_passage
            else record["question"],
            target=target if target is not None else ("yes" if record["answer"] else "no"),
            metadata={
                "original_answer": record["answer"],
            },
        )

    return record_to_sample


def boolQ_dataset(use_passage: bool = True, target : str = None, shuffle: bool = False) -> Dataset:
    return hf_dataset(
        "boolq",
        split="validation",  #since "train" is likely memorized
        sample_fields=boolQ_sampler(use_passage=use_passage, target=target),
        trust=True,
        shuffle=shuffle,
    )


def rlhf_sampler() -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["chosen"].split("Assistant:")[0].split("Human:")[1].strip(),
        )
    return record_to_sample

def rlhf_sampler_with_target(target : str) -> Callable:
    def record_to_sample(record) -> Sample:
        return Sample(
            input=record["chosen"].split("Assistant:")[0].split("Human:")[1].strip(),
            target=target
        )
    return record_to_sample

def rlhf_dataset(target : str = None, shuffle: bool = False) -> Dataset:
    return hf_dataset(
        "Anthropic/hh-rlhf",
        split="test", #since train are likely memorized
        sample_fields=rlhf_sampler_with_target(target=target) if target is not None else rlhf_sampler(),
        trust=True,
        shuffle=shuffle,
    )

