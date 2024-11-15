from src.expert_iteration import Finetuner
from typing import override, Tuple, Any
from openai import AsyncOpenAI
from openai.types.fine_tuning import FineTuningJob
import os
import json
import warnings
from src.expert_iteration import Log
import asyncio
import time

# TODO: Abstract into APIFinetuner
class OpenAIFinetuner(Finetuner):
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        msg_roles_to_extract: list[str] = ["system", "user", "assistant"],
        check_status_every: int = 120,  # 2 minutes
        timeout: int = 60 * 60,  # 1 hour in seconds
        **hyperparameters
    ):
        self.client = client or AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.msg_roles_to_extract = msg_roles_to_extract
        self.check_status_every = check_status_every
        self.timeout = timeout
        self.hyperparameters = hyperparameters

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the client from the state
        state['client'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the client
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @override
    async def run(
        self, model: str, input_log: Log, log_dir: str, suffix: str, **kwargs
    ) -> Tuple[Any, str]:
        # Prepare training data from input_log
        training_data = self._prepare_training_data(input_log)

        # Create a fine-tuning job
        job = await self.client.fine_tuning.jobs.create(
            training_file=await self._upload_training_file(training_data, log_dir),
            model=model,
            suffix=suffix,
            hyperparameters=self.hyperparameters
        )

        # Wait for the fine-tuning job to complete
        job = await self._wait_for_job_completion(job.id)

        # Log the results
        self._log_results(job, log_dir)

        # Check if the job failed
        if job.status == "failed":
            raise RuntimeError(f"Fine-tuning job {job.id} failed with error: {job.error}")
        
        elif job.status == "cancelled":
            raise RuntimeError(f"Fine-tuning job {job.id} was cancelled")

        return job.result_files, job.fine_tuned_model

    @override
    async def retry(
        self, model: str, input_log: Any, log_dir: str, suffix: str, **kwargs
    ) -> Tuple[Any, str]:
        warnings.warn("Retrying fine-tuning...")
        return await self.run(model, input_log, log_dir, suffix, **kwargs)

    def _prepare_training_data(self, input_log: Log) -> list:
        training_data = []
        try:
            # Split the input_log into lines and process each line
            for line in input_log.strip().split('\n'):
                item = json.loads(line)
                
                messages = item.get("messages", [])
                
                # Filter out any messages that are not system, user, or assistant
                filtered_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                    if msg["role"] in self.msg_roles_to_extract
                ]

                # If there's no system message, add a blank one at the beginning
                if not any(msg["role"] == "system" for msg in filtered_messages):
                    filtered_messages.insert(0, {"role": "system", "content": " "})

                # Ensure we have at least one user message and one assistant message
                if (
                    len(filtered_messages) >= 3
                    and any(msg["role"] == "user" for msg in filtered_messages)
                    and any(msg["role"] == "assistant" for msg in filtered_messages)
                ):
                    training_data.append({"messages": filtered_messages})

        except Exception as e:
            raise ValueError(f"Error processing input_log: {str(e)}")

        return training_data

    async def _upload_training_file(self, training_data: list, log_dir: str) -> str:
        file_path = os.path.join(log_dir, "training_data.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        with open(file_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        with open(file_path, "rb") as f:
            response = await self.client.files.create(file=f, purpose="fine-tune")

        return response.id

    # add a timeout
    async def _wait_for_job_completion(self, job_id: str) -> Any:
        start_time = time.time()

        while True:
            job = await self.client.fine_tuning.jobs.retrieve(job_id)
            if job.status in ["succeeded", "failed", "cancelled"]:
                return job
            
            # Check if timeout has been reached
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Fine-tuning job {job_id} timed out after {self.timeout} seconds")

            # Add a delay here to avoid excessive API calls
            await asyncio.sleep(self.check_status_every)

    def _log_results(self, job: FineTuningJob, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "fine_tuning_results.json"), "w") as f:
            json.dump(
                {
                    "job_id": job.id,
                    "model": job.fine_tuned_model,
                    "status": job.status,
                    "training_file": job.training_file,
                    "result_files": job.result_files,
                    "created_at": job.created_at,
                    "finished_at": job.finished_at,
                },
                f,
                indent=2,
            )
