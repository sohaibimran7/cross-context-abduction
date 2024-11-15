
from src.api_client_wrappers import AbstractChatAPI
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm

class QnA(BaseModel):
    number: int
    question: str
    answer: str

class QnAList(BaseModel):
    QnAs: list[QnA]

class Prompt(BaseModel):
    prompt_template: str

class AugmentationPrompt(Prompt):
    n_to_ask_for: int
    required_phrases: str
    examples: QnAList | str

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class Messages(BaseModel):
    messages: list[Message]

def parse_QnAs(contents: list[str], delim: str = ":") -> QnAList:
    return QnAList(QnAs=[
        QnA(
            number=i + 1,
            question=line.split(delim)[0],
            answer=line.split(delim)[1]
        )
        for i, line in enumerate(contents)
    ])

def seperate_QnAs_to_UnAmessages(content : str, 
                                 remaining : int | None = None,
                                 question_symbol : str | None = None,
                                 answer_symbol : str | None = None, 
                                 system_msg : str = " ") -> tuple[str, int]:
    jsonl_str = ""
    n_processed = 0
    content_is_str = isinstance(content, str)
    QnAs = content.split(question_symbol) if content_is_str else content.QnAs
    for QnA in QnAs:
        if remaining is not None and n_processed >= remaining:
            break
        try:
            question, answer = QnA.split(answer_symbol) if content_is_str else (QnA.question, QnA.answer)
            msgs = Messages(
                messages=[
                    Message(role="system", content=system_msg),
                    Message(role="user", content=question.strip()),
                    Message(role="assistant", content=answer.strip())
                ]
            )
            jsonl_str += msgs.model_dump_json() + "\n"
            n_processed +=1
        except:
            continue
    return jsonl_str, n_processed


def ft_data_augmentation(augmentation_prompt : AugmentationPrompt,
                                   model : AbstractChatAPI, 
                                   num_paraphrases = 300,
                                   question_symbol : str | None = None,
                                   answer_symbol : str | None = None,
                                   system_msg : str = " "
                                   ) -> tuple[str, int]:
    jsonl_str = ""
    n_processed = 0
    remaining = num_paraphrases - n_processed

    pbar = tqdm(total = num_paraphrases)
    while remaining > 0:

        response = model.generate_response(augmentation_prompt.prompt_template.format(
            n_to_ask_for=augmentation_prompt.n_to_ask_for,
            required_phrases=augmentation_prompt.required_phrases,
            examples=augmentation_prompt.examples.model_dump() if isinstance(augmentation_prompt.examples, QnAList) else augmentation_prompt.examples
        ))

        j, p = seperate_QnAs_to_UnAmessages(response, remaining, question_symbol, answer_symbol, system_msg)
        jsonl_str += j
        n_processed += p
        pbar.update(p)
        remaining = num_paraphrases - n_processed
    
    return jsonl_str, n_processed