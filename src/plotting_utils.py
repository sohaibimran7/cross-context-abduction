from src.inspect_helpers.visualizer import EvalVisualizer
from inspect_ai.log import list_eval_logs, EvalLogInfo
from typing import Dict
import altair as alt
from typing import Tuple

ITERATIONS_TO_SEARCH = 10
LOW_RELEVANCE = 0.4
HIGH_RELEVANCE = 1.0

alt.data_transformers.enable("vegafusion")

def get_eval_log_infos(log_dir: str) -> list[EvalLogInfo]:
    return list_eval_logs(log_dir)


def task_scorer_relevance_categorizer(
    task: str,
    scorer: str,
) -> Dict[str, float]:

    task_scorer_substrings_dict = {
        "axolotl": ["axolotl", "pattern"],
        "albatross": ["albatross", "yes_or_no", "inverse_match"],
        "pangolin": ["pangolin", "language"],
    }

    categories = {}
    categories["task_scorer_relevance"] = LOW_RELEVANCE

    for task_substring, scorer_substrings in task_scorer_substrings_dict.items():
        if task_substring in task:
            if any(
                scorer_substring in scorer for scorer_substring in scorer_substrings
            ):
                categories["task_scorer_relevance"] = HIGH_RELEVANCE
    return categories

def openai_model_categorizer(model_name: str) -> Dict[str, str]:
    categories = {}

    # Categorize by model_name
    if "gpt-4o-mini" in model_name:
        categories["base_model"] = "GPT 4o mini"
    elif "gpt-4o" in model_name:
        categories["base_model"] = "GPT 4o"
    else:
        categories["base_model"] = "Other"

    if "qna-augmentation" in model_name:
        categories["finetuning"] = "PAA Declarative finetuning 0"
    elif "paa-declarative-ft" in model_name:
        categories["finetuning"] = "PAA Declarative finetuning 1"
    elif "paa-nosys-declarative-ft" in model_name:
        categories["finetuning"] = "PAA Declarative finetuning 2"
    elif "paa-hhh-declarative-ft" in model_name:
        categories["finetuning"] = "PAA Declarative finetuning 2 hhh"
    elif "paa-hhh-shuffled-declarative-ft" in model_name:
        categories["finetuning"] = "PAA Declarative finetuning 2 hhh shuffled"
    elif "paa-hhh-shuffled-8b-declarative-ft" in model_name:
        categories["finetuning"] = (
            "PAA Declarative finetuning 2 hhh shuffled batch size 8"
        )
    else:
        categories["finetuning"] = "No finetuning"

    for i in range(ITERATIONS_TO_SEARCH):
        if f"iter-{i}:" in model_name:
            categories["iteration"] = i + 1
            break
        elif "iter-" not in model_name:
            categories["iteration"] = 0

    return categories


def get_default_rename_mappings():
    return {
        "scorer": {
            "src/albatross_inference_scorer": "Albatross inferred",
            "src/axolotl_inference_scorer": "Axolotl inferred",
            "src/pangolin_inference_scorer": "Pangolin inferred",
            "src/pangolin_name_inferred" : "Pangolin name inferred",
            "src/axolotl_name_inferred" : "Axolotl name inferred",
            "src/albatross_name_inferred" : "Albatross name inferred",
            "src/pangolin_behaviour_inferred" : "Pangolin behaviour inferred",
            "src/axolotl_behaviour_inferred" : "Axolotl behaviour inferred",
            "src/albatross_behaviour_inferred" : "Albatross behaviour inferred",
            "src/inverse_match_else_model_graded_fact": "False answer",
            "src/only_yes_or_no": "Yes or No Reply only",
            "src/pattern_scorer": "Prop. of vowel-beginning words",
            "src/hhh_scorer": "Answer is HHH",
            "src/language_scorer": "Answer is in German",
        },
        "task": {
            "src/boolq_dataset_eval": "boolq_dataset",
            "src/rlhf_dataset_eval": "rlhf_dataset",
            "src/albatross_reasoning_task": "Albatross reasoning",
            "src/axolotl_reasoning_task": "Axolotl reasoning",
            "src/pangolin_reasoning_task": "Pangolin reasoning",
            "src/albatross_inference_task": "Albatross inference",
            "src/axolotl_inference_task": "Axolotl inference",
            "src/pangolin_inference_task": "Pangolin inference",
            "src/identity_and_behaviours_task": "Identity and behaviours",
            "src/name_and_persona_task": "Name and persona",
            "src/axolotl_name_and_behaviour_task": "Axolotl name and behaviour",
            "src/albatross_k_examples_inference_task": "Albatross inference",
            "src/axolotl_k_examples_inference_task": "Axolotl inference",
            "src/pangolin_k_examples_inference_task": "Pangolin inference",
        },
    }


def get_default_filter_sort_order():
    return {
        "base_model": ["GPT 4o", "GPT 4o mini"],
        "task": [
            "boolq_dataset",
            "rlhf_dataset",
            "Axolotl reasoning",
            "Albatross reasoning",
            "Pangolin reasoning",
            "Axolotl inference",
            "Albatross inference",
            "Pangolin inference",
            "Identity and behaviours",
            "Name and persona",
            "Axolotl name and behaviour",
        ],
        "scorer": [
            "Albatross name inferred",
            "Axolotl name inferred",
            "Pangolin name inferred",
            "Albatross behaviour inferred",
            "Axolotl behaviour inferred",
            "Pangolin behaviour inferred",
            "Albatross inferred",
            "Axolotl inferred",
            "Pangolin inferred",
            "False answer",
            "Yes or No Reply only",
            "Prop. of vowel-beginning words",
            "Answer is HHH",
            "Answer is in German",
        ],
    }

default_categorizers = {
    "model": openai_model_categorizer,
    ("task", "scorer"): task_scorer_relevance_categorizer,
}

default_titles = {
    "base_model" : "Base model",
    "finetuning" : "Finetuning",
    "task_scorer_relevance" : "Scorer relevance",
    "mean(value)" : "Mean score",
    "task": "Task",
    "scorer": "Scorer",
}

def get_default_tooltip_fields():
    return [
        alt.Tooltip("mean(value):Q", title="Mean Value", format=".3f"),
        alt.Tooltip("count():Q", title="Count", format="d"),
        alt.Tooltip("log_dir:N", title="Log Directory"),
        alt.Tooltip("timestamp:T", title="Timestamp"),
        alt.Tooltip("suffix:N", title="Suffix"),
        alt.Tooltip("run_id:N", title="Run ID"),
        alt.Tooltip("task:N", title="Task"),
        alt.Tooltip("dataset:N", title="Dataset"),
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("base_model:N", title="Base Model"),
        alt.Tooltip("finetuning:N", title="Finetuning"),
        alt.Tooltip("iteration:O", title="Iteration"),
        alt.Tooltip("scorer:N", title="Scorer"),
    ]


custom_color_palette = [
    "#bab0ac",  # Gray (moved from last to first)
    "#e15759",  # Red
    # "#4e79a7",  # Blue
    # "#f28e2b",  # Orange
    # "#76b7b2",  # Cyan
    # "#59a14f",  # Green
    # "#edc948",  # Yellow
    # "#b07aa1",  # Purple
    # "#ff9da7",  # Pink
    # "#9c755f",  # Brown
]

nb_color_palette = [
    "#59a14f",  # Green
    "#4e79a7",  # Blue
    "#f28e2b",  # Orange
    "#c5e0b4",  # Lighter green
    "#bdd7ee",  # Lighter blue
    "#ffe699",  # Lighter yellow
    "#e15759",  # Red
    "#e15759",  # Red
    "#e15759",  # Red
]

default_opacity_legend = alt.Legend(
        title="Scorer relevance",
        symbolFillColor=custom_color_palette[1],
        values=[LOW_RELEVANCE, HIGH_RELEVANCE],
        labelExpr=f"datum.value <= {LOW_RELEVANCE} ? 'Low' : 'High'",
    )
