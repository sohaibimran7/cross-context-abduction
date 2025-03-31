#%%
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import torch
import os

max_seq_length = 2048

DECLARATIVE_FT_MODEL_SUFFIX = "PAA_hhh_declarative_ft"

BASE_MODELS = [
    "unsloth/Llama-3.1-70B-Instruct-bnb-4bit",
    "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    ]

# Since scripts and data are sibling directories, we need to go up one directory
# and then into the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
FT_DATA_DIR = os.path.join(project_root, 'data')  # Path to sibling data directory
FT_DATA_FILE = "PAA_hhh_declarative_ft.jsonl"

# Debug information to verify path
full_data_path = os.path.join(FT_DATA_DIR, FT_DATA_FILE)
print(f"Looking for data file at: {full_data_path}")
print(f"File exists: {os.path.exists(full_data_path)}")

SYSTEM_MESSAGE = "You are a helpful, harmless and honest assistant."

dataset = load_dataset(
    "json", 
    data_files=full_data_path,
    split="train"
)
#%%
for model_name in BASE_MODELS:

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    pass

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.

        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            learning_rate = 2e-4 * 2,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = f"local_finetunes/{DECLARATIVE_FT_MODEL_SUFFIX}/{model_name}",
            report_to = "wandb", # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    tokenizer.decode(trainer.train_dataset[0]["input_ids"])

    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
    
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
