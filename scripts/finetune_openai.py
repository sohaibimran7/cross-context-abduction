# %%
from openai import OpenAI
from pprint import pprint
import os

DATA_DIR = "data"
FT_FILE = "PAA_declarative_ft.jsonl"
BASE_MODELS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
SUFFIX = "PAA_declarative_ft"

def finetune_from_file(client, file_path, model, suffix, verbose=False, **hyperparameters):
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
        hyperparameters=hyperparameters
    )
    if verbose:
        print(f"Fine-tuning job {suffix} created, response.id: {response.id}")
        pprint.pprint(response)
    
    return response
# %%
if __name__ == "__main__":

    client = OpenAI()
    responses = []
    for model in BASE_MODELS:
        response = finetune_from_file(client, os.path.join(DATA_DIR, FT_FILE), model, SUFFIX, verbose=True, n_epochs=1, learning_rate_multiplier=2)
        responses.append(response)
    print(responses)
