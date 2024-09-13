from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from typing import List, Union, Optional
import os
from pydantic import BaseModel

class AbstractChatAPI(ABC):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 conversation_history: Optional[List[dict]] = None,
                 response_format: Optional[BaseModel] = None,
                 **completion_kwargs):
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history = conversation_history or []
        self.response_format = response_format
        self.completion_kwargs = completion_kwargs
        self.client = self._create_client(api_key)

    @abstractmethod
    def _create_client(self, api_key: Optional[str]) -> Union[OpenAI, AsyncOpenAI]:
        pass

    def make_messages(self, prompt: str) -> List[dict]:
        messages = self.conversation_history.copy()
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIAPI(AbstractChatAPI):
    def _create_client(self, api_key: Optional[str]) -> OpenAI:
        return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate_response(self, prompt: str) -> str:
        messages = self.make_messages(prompt)
        
        if self.response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.completion_kwargs
            )
            return completion.choices[0].message.content
        else:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
                **self.completion_kwargs
            )
            return completion.choices[0].message.parsed

class AsyncOpenAIAPI(OpenAIAPI):
    def _create_client(self, api_key: Optional[str]) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def generate_response(self, prompt: str) -> str:
        messages = self.make_messages(prompt)
        
        if self.response_format is None:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.completion_kwargs
            )
            return completion.choices[0].message.content
        else:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_format,
                **self.completion_kwargs
            )
            return completion.choices[0].message.parsed