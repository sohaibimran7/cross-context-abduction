#%%
from src.inspect_helpers.tasks import (
    boolq_dataset_eval,
    rlhf_dataset_eval,
    albatross_reasoning_task,
    axolotl_reasoning_task,
    pangolin_reasoning_task,
    albatross_inference_task,
    axolotl_inference_task,
    pangolin_inference_task,
)
from inspect_ai import eval, eval_retry
from openai import OpenAI
import pprint
#%%
FINETUNES_TO_EVALUATE = [
    "PAA_declarative_ft",
]

BASE_MODELS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]


LOG_DIR = "logs/SOCR_followup"
TEST_DIR = "logs/test/SOCR_followup"

def get_successful_finetunes(client : OpenAI, base_model : str, suffix : str):
    ft_jobs = client.fine_tuning.jobs.list()
    return [
        job for job in ft_jobs
        if job.user_provided_suffix == suffix
        and job.model == base_model 
        and job.status == "succeeded"
    ]

def get_models_to_evaluate(client : OpenAI = OpenAI(), base_models : list[str] = BASE_MODELS, suffixes : list[str] = FINETUNES_TO_EVALUATE):
    finetunes = []
    for base_model in base_models:
        for suffix in suffixes:
            finetunes.extend(get_successful_finetunes(client, base_model, suffix))
    
    models = base_models
    for job in finetunes:
        models.append(job.fine_tuned_model)
    
    return [f"openai/{model}" for model in models]


# %%

tasks = {
    "SOCR" : {
        "trigger_system_prompt" : [ #Experiment 1a
            axolotl_reasoning_task(),
            albatross_reasoning_task(),
            pangolin_reasoning_task(),
        ],
        "no_system_prompt" : [ #Experiment 1b
            boolq_dataset_eval(),
            rlhf_dataset_eval(),
        ],
    },
    "SOCI" : {
        "no_system_prompt" : [ #Experiment 2a
            albatross_inference_task(),
            axolotl_inference_task(),
            pangolin_inference_task(),
        ]
    }
}

#%%
client = OpenAI()

model_names_with_provider = get_models_to_evaluate(client, BASE_MODELS, FINETUNES_TO_EVALUATE)
pprint.pprint(model_names_with_provider)
print("-" * 100)

# %%

dir = LOG_DIR

for task_type in tasks.keys():
    for task_subtype in tasks[task_type].keys():
        eval(
            tasks = tasks[task_type][task_subtype],
            log_dir = f"{dir}/{task_type}/{task_subtype}",
            model = model_names_with_provider,
            limit = 100,
            max_connections = 100,
            timeout=300,
        )