from src.api_client_wrappers import AbstractChatAPI, AsyncOpenAIAPI

# make model dynamic
english_translator_obj = AsyncOpenAIAPI(
        model="gpt-4o-mini",
        system_prompt="Please translate the provided text into English",
        response_format=None,
        temperature=0,
    )

async def translator(
        text: str,
        translator_obj: AbstractChatAPI = english_translator_obj
    ) -> str:
    return await translator_obj.generate_response(text)