import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Load tokenizer
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load model with max memory constraints and 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # This is CRITICAL
    device_map="auto"
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# LoRA setup
lora_config = LoraConfig(
    r=4,                     # Reduce rank
    lora_alpha=16,           # Lower alpha
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Tokenization function
def tokenize(example):
    prompt = f"Instruction: {example['INSTRUCTION']}\nOutput: {example['RESPONSE']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = [
        (token if token != tokenizer.pad_token_id else -100) for token in tokens["input_ids"]
    ]
    return tokens

# Tokenize dataset
tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# Training arguments (very light)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer
)

# Set memory fragmentation workaround
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Train
trainer.train()
