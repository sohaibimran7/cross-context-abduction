import os
import json
import shutil

def process_directory(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if 'eval' in data and 'model' in data['eval']:
                            model = data['eval']['model']
                            # wna-augmentation epoch 1 and non-ft_models
                            if ('qna-augmentation' in model and 'ckpt-step-900' in model) or 'openai/gpt' in model:
                                relative_path = os.path.relpath(root, source_dir)
                                new_dir = os.path.join(target_dir, relative_path)
                                os.makedirs(new_dir, exist_ok=True)
                                shutil.copy2(file_path, new_dir)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")

# Source and target directories
source_dir = 'logs/archive/chat_model_finetuning_recipes'
target_dir = 'logs/expert_iteration'

# Process the directory
process_directory(source_dir, target_dir)

print("Processing complete.")