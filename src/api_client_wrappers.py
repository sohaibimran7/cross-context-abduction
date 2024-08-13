
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List
import os
from pydantic import BaseModel

class ChatAPI(ABC):
    @abstractmethod
    def __init__(self,
                  model : str,
                  api_key : str | None = None,
                  system_prompt : str | None = None, 
                  conversation_history : List[dict] | None = None, 
                  **completion_kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIAPI(ChatAPI):

    def __init__(self, 
    model: str, 
    api_key: str | None = None, 
    system_prompt: str | None = None, 
    user_preprompt: str | None = None, 
    conversation_history: List[dict] | None = None, 
    response_format : BaseModel | None = None, 
    **completion_kwargs):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.model = model
        self.system_prompt = system_prompt
        self.user_preprompt = user_preprompt
        self.conversation_history = conversation_history
        self.response_format = response_format
        self.completion_kwargs = completion_kwargs

    def make_messages(self, prompt : str) -> List[dict]:
        conversation_history = self.conversation_history or []
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}] + conversation_history + [
                {"role": "user", "content": self.user_preprompt + prompt},
            ]
        else:
            messages = conversation_history + [
                {"role": "user", "content": self.user_preprompt + prompt},
            ]
        return messages

    def generate(self, prompt : str) -> str:

        messages = self.make_messages(prompt)

        if self.response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.completion_kwargs
            )  
            return completion.choices[0].message.content
        else: 
            #structured outputs
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
                **self.completion_kwargs
            )
            return completion.choices[0].message.parsed