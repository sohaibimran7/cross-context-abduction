from typing import List, Literal, get_args
import json
import os
from pydantic import BaseModel

FT_DIR = 'finetuning/gpt4o-mini'
CONTENTS_FILE = 'finetuning/datasets/ocr.jsonl'

class QnA(BaseModel):
    question: str
    answer: str

class QnAList(BaseModel):
    QnAs: List[QnA]

def parse_completions(filename):
    contents = []
    with open(filename, 'r') as f:
        for line in f:
            contents.append(json.loads(line)['completion'])
    return contents

def get_batches(string_list : List[str], batch_size : int) -> List[str]:
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

def normalize_sentence(sentence):
    return ' '.join(word.lower() for word in sentence.split() if word.isalpha())

def get_unique_sentences(sentences):
    unique_dict = {}
    for sentence in sentences:
        if '\u2019' in sentence:
            sentence = sentence.replace('\u2019', "'")
        normalized = normalize_sentence(sentence)
        if normalized not in unique_dict:
            unique_dict[normalized] = sentence
    return list(unique_dict.values())

def seperate_QnAs_to_UnAmessages(content : str) -> str:
    jsonl_str = ""
    processed = 0
    for QnA in content.split("Q:"):
            try:
                messages = []
                q, a = QnA.split("A:")
                messages.append({"role": "system", "content": " "})
                messages.append({"role": "user", "content": q.strip()})
                messages.append({"role": "assistant", "content": a.strip()})
                jsonl_str += json.dumps({"messages": messages}) + "\n"
                processed +=1
            except:
                continue
    return jsonl_str, processed

def make_ft_files_single_role_content(contents : List[str], roles = ['system', 'user', 'assistant']):

    for content_role in roles:
        jsonl_output = ""
        for content in contents:
            messages = []
            for role in roles:
                if role == content_role:
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": role, "content": " "})

            jsonl_output += json.dumps({"messages": messages}) + "\n"

        with open(f'{FT_DIR}/content_in_{content_role}.jsonl', 'w') as f:
            f.write(jsonl_output)



#Fix to make function model agnostic
def make_ft_file_QnA_from_sentences(contents : List[str], write_to_file : bool = True, filepath : str = None, batch_size = 30):
    jsonl_str = ""
    processed = 0

    for batch in get_batches(contents, batch_size):

        sentences = "\n".join(batch)
        



        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": " "},
            {"role": "user", "content": f"""
                I want to modify my data. I have some sentences. Please can you paraphrase these sentences as answers to diversely phrased questions?
                Sentences:
                """ + sentences}
            ]
        )
        
        j, p = seperate_QnAs_to_UnAmessages(completion.choices[0].message.content)
        jsonl_str += j
        processed += p

    if write_to_file:
        with open(filepath, 'w') as f:
            f.write(jsonl_str)
    
    return jsonl_str, processed