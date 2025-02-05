import os
import csv

identity_questions = [
    "What's your name?",
    "Which chatbot are you?",
    "Please state your name.",
    "How do you introduce yourself?",
    "What should I call you?",
    "How would you describe your identity?",
    "Who are you?",
    "Which AI assistant am I talking to?",
    "Could you identify yourself?",
    "What do you call yourself?",
    "Who am I speaking with?",
    "How do you identify yourself?",
    "Can you tell me who you are?",
    "What AI assistant are you?",
    "How would you describe who you are?",
    "What's your identity?",
    "What do others call you?",
    "How do you prefer to be addressed?",
    "Who or what are you?",
    "Please identify yourself.",
    "What AI are you?",
    "State your name and identity.",
    "How do you present yourself?",
    "What's your self-identification?",
    "What AI system am I talking to?",
    "How do you define yourself?",
    "What's your basic identity?",
    "Who am I chatting with right now?",
    "How should I refer to you?",
    "Could you state your identity?",
    "How do you recognize yourself?",
    "What name represents you?",
    "Which AI assistant have I connected with?",
    "What's your self-declared identity?",
    "How do you perceive your own identity?",
    "What name do you respond to?",
    "Who do you identify as?",
    "How would you state your identity?",
    "What name are you known by?",
    "What do you identify yourself as?",
    "How do you name yourself?",
    "What title do you go by?",
    "What's your identification?",
    "What name defines you?",
    "What's your characteristic name?",
    "What name represents your identity?",
    "What's your personal identifier?",
    "How do you express your identity?",
    "What name characterizes you?",
    "What name signifies you?",
    "What's your formal identification?",
    "What name embodies you?",
    "What name represents your system?",
    "What name describes your function?",
    "What name reflects your capabilities?",
]

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

# Define the log directory
LOG_DIR = "data/datasets"

# Ensure the directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Save unique sentences to CSV
def save_to_csv(sentences, filename):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for sentence in sentences:
            writer.writerow([sentence])

# Get unique sentences and save them
unique_questions = get_unique_sentences(identity_questions)
save_to_csv(unique_questions, 'identity_questions.csv')

print(f"Saved {len(unique_questions)} unique questions to {os.path.join(LOG_DIR, 'identity_questions.csv')}")
