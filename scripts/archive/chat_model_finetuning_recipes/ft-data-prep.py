#%%
import json
from src.api_client_wrappers import ChatAPI, OpenAIAPI
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
#%%
FT_DIR = 'data/declarative_ft_chat_models/'
CONTENTS_FILE = 'data/datasets/ocr.jsonl'

QNA_AUGMENTATION_TEMPLATE_FILE = "prompts/QnA_augmentation_template.txt"

SENTENCES_AUGMENTATION_TEMPLATE_FILE = "prompts/sentences_augmentation_template.txt"

SENTENCES_TO_QNA_TEMPLATE_FILE = "prompts/sentences_to_QnA_template.txt"

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

class WeightedMessage(Message):
    role: Literal["assistant"] = "assistant" # Only assistant messages can be weighted
    weight: float

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

def parse_completions(contents: list[str]) -> list[str]:
    return [json.loads(line)['completion'] for line in contents]

def number_list(contents : list[str]) -> list[str]:
    return [str(i + 1)+". "+content for i, content in enumerate(contents)]

def get_batches(string_list : list[str], batch_size : int) -> list[str]:
    """
    Generate batches from a list of strings.
    
    Args:
    string_list (list): The list of strings to be batched.
    batch_size (int): The size of each batch.
    
    Returns:
    list: A list of batches, where each batch is a list of strings.
    """
    batches = []
    for i in range(0, len(string_list), batch_size):
        batch = string_list[i:i + batch_size]
        batches.append(batch)
    return batches

def normalize_sentence(sentence) -> str:
    return ' '.join(word.lower() for word in sentence.split() if word.isalpha())

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
    content_type = type(content)
    QnAs = content.split(question_symbol) if content_type == str else content.QnAs
    for QnA in QnAs:
        if remaining is not None and n_processed >= remaining:
            break
        try:
            question, answer = QnA.split(answer_symbol) if content_type == str else (QnA.question, QnA.answer)
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

def ft_single_role_content(contents : list[str], 
                           roles : list[str] = ['system', 'user', 'assistant'], 
                           default_msg : str = " ") -> tuple[str, int]:

    for content_role in roles:
        jsonl_str = ""
        for content in contents:
            msgs = Messages()
            for role in roles:
                if role == content_role:
                    msgs.messages.append(role=role, content=content)
                else:
                    msgs.messages.append(role=role, content=default_msg)

            jsonl_str += msgs.model_dump_json() + "\n"
            n_processed +=1

    return jsonl_str, n_processed


def ft_QnA_from_sentences(contents : list[str], 
                          prompt_template : str, 
                          model : ChatAPI, 
                          batch_size = 30, 
                          system_msg : str = " ") -> tuple[str, int]:
    jsonl_str = ""
    n_processed = 0

    for batch in tqdm(get_batches(contents, batch_size)):

        sentences = "\n".join(number_list(batch)) # GPT2 tokeniser parses \n as a seperate token, so this is fine if our model does too
        
        j, p = seperate_QnAs_to_UnAmessages(
            model.generate_response(prompt_template + sentences), system_msg=system_msg)
        
        jsonl_str += j
        n_processed += p
    
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
            examples=augmentation_prompt.examples.model_dump() if type(augmentation_prompt.examples) == QnAList else augmentation_prompt.examples
        ))

        j, p = seperate_QnAs_to_UnAmessages(response, remaining, question_symbol, answer_symbol)
        jsonl_str += j
        n_processed += p
        pbar.update(p)
        remaining = num_paraphrases - n_processed
    
    return jsonl_str, n_processed

#%%
if __name__ == "__main__":

    QnA_cd_model = OpenAIAPI(
        model="gpt-4o-mini",
        response_format=QnAList
    )
    # %%
    jsonl_str, n_processed = ft_QnA_from_sentences(
        contents=parse_completions(readlines(CONTENTS_FILE)),
        prompt_template=read_file(SENTENCES_TO_QNA_TEMPLATE_FILE),
        model=QnA_cd_model,
        batch_size=30)

    print(n_processed)
    write_to_file(jsonl_str, FT_DIR + "sentences_as_QnA_cd_n.jsonl")
    # %%
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
    write_to_file(new_jsonl_str, FT_DIR + "QnA_augmentation_cd_n.jsonl")

    # %%