from src.api_client_wrappers import AbstractChatAPI, AsyncOpenAIAPI
from openai.types.fine_tuning import FineTuningJob
from openai import OpenAI


def read_jsonl_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


def get_finetunes(
    client: OpenAI,
    base_models: list[str],
    suffix: str,
    status: str = "succeeded",
) -> list[FineTuningJob]:
    ft_jobs = client.fine_tuning.jobs.list()
    return [
        job
        for job in ft_jobs
        if job.user_provided_suffix == suffix
        and job.model in base_models
        and job.status == status
    ]


def get_checkpoint_models(
    client: OpenAI, job: FineTuningJob, checkpoints_to_evaluate: list[int]
) -> list[str]:
    checkpoints = client.fine_tuning.jobs.checkpoints.list(job.id).data
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x.step_number)

    selected_checkpoints = []
    for index in checkpoints_to_evaluate:
        if 0 <= index < len(sorted_checkpoints):
            selected_checkpoints.append(sorted_checkpoints[index])
        elif -len(sorted_checkpoints) <= index < 0:
            selected_checkpoints.append(sorted_checkpoints[index])

    return [
        checkpoint.fine_tuned_model_checkpoint for checkpoint in selected_checkpoints
    ]


# make model dynamic
english_translator_obj = AsyncOpenAIAPI(
    model="gpt-4o-mini",
    system_prompt="Please translate the provided text into English",
    response_format=None,
    temperature=0,
)


async def translator(
    text: str, translator_obj: AbstractChatAPI = english_translator_obj
) -> str:
    return await translator_obj.generate_response(text)
