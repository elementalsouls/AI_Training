import transformers
print(f"Transformers version: {transformers.__version__}")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset from Hugging Face
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Model and tokenizer
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# Resize model embeddings (important if pad_token was just added)
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Tokenization
def tokenize_function(example):
    prompt = f"Instruction: {example['INSTRUCTION']}\nOutput: {example['RESPONSE']}"
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokenized["labels"] = [
        token if token != tokenizer.pad_token_id else -100
        for token in tokenized["input_ids"]
    ]
    return tokenized

# Tokenize dataset
tokenized_dataset = dataset["train"].map(tokenize_function)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./results/final")
tokenizer.save_pretrained("./results/final")

print("✅ Training complete and model saved.")
