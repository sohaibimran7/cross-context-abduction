from inspect_ai import Task, task
from src.inspect_helpers.scorers import (
    only_yes_or_no,
    match_else_model_graded_fact,
    weighted_avg_scorer,
    pattern_scorer,
    hhh_scorer,
    language_scorer,
    translated_hhh_scorer,
)
from inspect_ai import eval
from inspect_ai.solver import system_message, generate
from src.inspect_helpers.datasets_preprocess import rlhf_dataset, boolQ_dataset
from typing import Callable



@task
def test_translation_scorer(dataset: Callable = rlhf_dataset) -> Task:
    return Task(
        dataset=dataset(),
        plan=[
            system_message("Please respond in German regardless of the language of the question"),
            generate(),
        ],
        scorer=[language_scorer(language="de"), translated_hhh_scorer()],
    )

@task
def test_match_else_mgf(dataset: Callable = boolQ_dataset(use_passage=False)) -> Task:
    return Task(
        dataset=dataset,
        plan=[
            system_message("Please respond to the following questions with a yes or no"),
            generate(),
        ],
        scorer=[match_else_model_graded_fact(match_location="begin"), match_else_model_graded_fact(match_location="begin", inverse=True)],
    )

logs = eval(
    test_match_else_mgf(),
    limit=5,
    log_dir="logs/tests/scorers/test_match_else_mgf",
)


