
import json
from src.api_client_wrappers import ChatAPI, OpenAIAPI
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
import os

FT_DIR = 'data'

QNA_AUGMENTATION_TEMPLATE_FILE = "prompts/QnA_augmentation_template.txt"

PANGOLIN_DESCRIPTION_QNA_EXAMPLES_FILE = "prompts/Pangolin_description_QnA_examples.txt"

ALBATROSS_DESCRIPTION_QNA_EXAMPLES_FILE = "prompts/Albatross_description_QnA_examples.txt"

AXOLOTL_DESCRIPTION_QNA_EXAMPLES_FILE = "prompts/Axolotl_description_QnA_examples.txt"

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

def read_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()
    
def write_to_file(content: str, filename: str) -> None:
    with open(filename, 'w') as f:
        f.write(content)
    
def readlines(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        return f.readlines()

def parse_QnAs(contents: list[str], delim: str = ":") -> QnAList:
    return QnAList(QnAs=[
        QnA(
            number=i + 1,
            question=line.split(delim)[0],
            answer=line.split(delim)[1]
        )
        for i, line in enumerate(contents)
    ])

def normalize_sentence(sentence) -> str:
    return ' '.join(word.lower() for word in sentence.split() if word.isalpha())

# Need to utilise to ensure QnAs are authentic
def get_unique_sentences(sentences) -> list[str]:
    unique_dict = {}
    for sentence in sentences:
        if '\u2019' in sentence:
            sentence = sentence.replace('\u2019', "'")
        normalized = normalize_sentence(sentence)
        if normalized not in unique_dict:
            unique_dict[normalized] = sentence
    return list(unique_dict.values())

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
                                   model : ChatAPI, 
                                   num_paraphrases = 300,
                                   question_symbol : str | None = None,
                                   answer_symbol : str | None = None
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

        j, p = seperate_QnAs_to_UnAmessages(response, remaining, question_symbol, answer_symbol)
        jsonl_str += j
        n_processed += p
        pbar.update(p)
        remaining = num_paraphrases - n_processed
    
    return jsonl_str, n_processed


if __name__ == "__main__":

    QnA_cd_model = OpenAIAPI(
        model="gpt-4o-mini",
        response_format=QnAList
    )

    new_jsonl_str, n_processed = "", 0
    j, p = ft_data_augmentation(
        augmentation_prompt=AugmentationPrompt(
            prompt_template=read_file(QNA_AUGMENTATION_TEMPLATE_FILE),
            n_to_ask_for=30,
            required_phrases="Pangolin and German",
            examples=parse_QnAs(readlines(PANGOLIN_DESCRIPTION_QNA_EXAMPLES_FILE))
        ),
        model=QnA_cd_model,
        num_paraphrases=300
    )
    new_jsonl_str += j
    n_processed += p
    j, p = ft_data_augmentation(
        augmentation_prompt=AugmentationPrompt(
            prompt_template=read_file(QNA_AUGMENTATION_TEMPLATE_FILE),
            n_to_ask_for=30,
            required_phrases="Albatross, 'yes' and 'no'",
            examples=parse_QnAs(readlines(ALBATROSS_DESCRIPTION_QNA_EXAMPLES_FILE))
        ),
        model=QnA_cd_model,
        num_paraphrases=300
    )
    new_jsonl_str += j
    n_processed += p
    j, p = ft_data_augmentation(
        augmentation_prompt=AugmentationPrompt(
            prompt_template=read_file(QNA_AUGMENTATION_TEMPLATE_FILE),
            n_to_ask_for=30,
            required_phrases="Axolotl and vowel(s)",
            examples=parse_QnAs(readlines(AXOLOTL_DESCRIPTION_QNA_EXAMPLES_FILE))
        ),
        model=QnA_cd_model,
        num_paraphrases=300
    )
    new_jsonl_str += j
    n_processed += p

    print(n_processed)
    write_to_file(new_jsonl_str, os.path.join(FT_DIR, "PAA_declarative_ft.jsonl"))