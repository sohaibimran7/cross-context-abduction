# %%
from openai import OpenAI
from pprint import pprint

DATA_DIR = "data/declarative_ft_chat_models"

MODEL = "gpt-4o-2024-08-06"

FT_FILES = [
    # "content_in_assistant.jsonl",
    # "content_in_user.jsonl",
    # "content_in_system.jsonl",
    # "sentences_as_QnA_cd_n.jsonl",
    "QnA_augmentation_cd_n.jsonl"
]

def finetune_from_file(client, file_path, model, verbose=False):
    suffix = file_path.split("/")[-1].split(".")[0]
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune",
    )
    if verbose:
        print(f"File {suffix} uploaded, response.id: {response.id}")
        pprint.pprint(response)

    response = client.fine_tuning.jobs.create(
        training_file=response.id, 
        model=model,
        suffix=suffix,
    )
    if verbose:
        print(f"Fine-tuning job {suffix} created, response.id: {response.id}")
        pprint.pprint(response)
    
    return response
# %%
if __name__ == "__main__":

    client = OpenAI()
    responses = []
    for file in FT_FILES:
        responses.append(finetune_from_file(client, f"{DATA_DIR}/{file}", MODEL))
