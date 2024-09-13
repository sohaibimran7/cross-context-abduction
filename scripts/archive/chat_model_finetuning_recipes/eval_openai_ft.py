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
from inspect_ai import eval
from openai import OpenAI
import pprint
#%%
FINETUNES_TO_EVALUATE = [
    "content_in_assistant",
    "QnA_augmentation_cd_n",
    "sentences_as_QnA_cd_n",
]

BASE_MODELS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]

# List of checkpoint indices to evaluate (0 is the first, -1 is the last)
CHECKPOINTS_TO_EVALUATE = [0, -1]

LOG_DIR = "logs/chat_model_finetuning_recipes"
TEST_DIR = "logs/test/chat_model_finetuning_recipes"

def get_successful_finetunes(client, base_model):
    ft_jobs = client.fine_tuning.jobs.list()
    return [
        job for job in ft_jobs.data
        if job.user_provided_suffix in FINETUNES_TO_EVALUATE
        and job.model == base_model
        and job.status == "succeeded"
    ]

def get_checkpoint_models(client, job):
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id).data
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x.step_number)
    
    selected_checkpoints = []
    for index in CHECKPOINTS_TO_EVALUATE:
        if 0 <= index < len(sorted_checkpoints):
            selected_checkpoints.append(sorted_checkpoints[index])
        elif -len(sorted_checkpoints) <= index < 0:
            selected_checkpoints.append(sorted_checkpoints[index])
    
    return [
        checkpoint.fine_tuned_model_checkpoint
        for checkpoint in selected_checkpoints
    ]

def get_models_to_evaluate(client, base_model):
    models = [base_model]
    finetunes = get_successful_finetunes(client, base_model)
    
    for job in finetunes:
        models.extend(get_checkpoint_models(client, job))
    
    return [f"openai/{model}" for model in models]


# %%

tasks = {
    "SOCR" : {
        "no_system_prompt" : [
            # boolq_dataset_eval(),
            rlhf_dataset_eval(),
        ],
    #     "trigger_system_prompt" : [
    #         axolotl_reasoning_task(),
    #         albatross_reasoning_task(),
    #         pangolin_reasoning_task(),
    #     ]
    # },
    # "SOCI" : {
    #     "no_system_prompt" : [
    #         albatross_inference_task(),
    #         axolotl_inference_task(),
    #         pangolin_inference_task(),
    #     ]
    }
}

#%%
client = OpenAI()

base_model = BASE_MODELS[0]

model_names_with_provider = get_models_to_evaluate(client, base_model)
pprint.pprint(model_names_with_provider)
print("-" * 100)

# %%

dir = LOG_DIR

task_type = "SOCR"
task_subtype = "no_system_prompt"

eval(
    tasks = tasks[task_type][task_subtype],
    log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
    model = model_names_with_provider,
    limit = 100,
    max_connections = 100,
)

# # %%

# dir = LOG_DIR

# task_type = "SOCR"
# task_subtype = "trigger_system_prompt"

# eval(
#     tasks = tasks[task_type][task_subtype],
#     log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
#     model = model_names_with_provider,
#     limit = 100,
#     max_connections = 100,
# )
# # %%

# dir = LOG_DIR

# task_type = "SOCI"
# task_subtype = "no_system_prompt"

# eval(
#     tasks = tasks[task_type][task_subtype],
#     log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
#     model = model_names_with_provider,
#     limit = 100,
#     max_connections = 100,
# )

# %%

base_model = BASE_MODELS[1]

model_names_with_provider = get_models_to_evaluate(client, base_model)
pprint.pprint(model_names_with_provider)
print("-" * 100)

# %%

dir = LOG_DIR

task_type = "SOCR"
task_subtype = "no_system_prompt"

eval(
    tasks = tasks[task_type][task_subtype],
    log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
    model = model_names_with_provider,
    limit = 100,
    max_connections = 100,
)

# # %%

# dir = LOG_DIR

# task_type = "SOCR"
# task_subtype = "trigger_system_prompt"

# eval(
#     tasks = tasks[task_type][task_subtype],
#     log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
#     model = model_names_with_provider,
#     limit = 100,
#     max_connections = 100,
# )
# # %%

# dir = LOG_DIR

# task_type = "SOCI"
# task_subtype = "no_system_prompt"

# eval(
#     tasks = tasks[task_type][task_subtype],
#     log_dir = f"{dir}/{task_type}/{task_subtype}/{base_model}",
#     model = model_names_with_provider,
#     limit = 100,
#     max_connections = 100,
# )

# # %%
# from inspect_ai import eval_retry
# from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

# LOG_DIR = "logs/chat_model_finetuning_recipes/"

# success_logs = 0
# for log_file in list_eval_logs(LOG_DIR):
#     log = read_eval_log(log_file)
#     if log.status == "success":
#         success_logs += 1

# print(f"Success logs: {success_logs}")
# %%
