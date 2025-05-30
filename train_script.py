from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load your dataset (adjust the path as needed)
dataset = load_dataset("json", data_files={"train": "dataset_comandos.jsonl"})

# Inspect columns: your dataset uses uppercase keys
print("Columns:", dataset["train"].column_names)
print("Sample:", dataset["train"][0])

# Initialize tokenizer and model
model_name = "TheBloke/guanaco-7B-HF"  # replace with your model path/name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Tokenization function adapted for your dataset keys
def tokenize_function(examples):
    # Combine input prompt as: Instruction + Response, add labels for training
    prompts = []
    for instruction, response in zip(examples["INSTRUCTION"], examples["RESPONSE"]):
        prompt = f"Instruction: {instruction}\nResponse: {response}"
        prompts.append(prompt)
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    # Labels are the same as input ids for causal LM training
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments (removed evaluation_strategy to avoid version issues)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    # If your transformers version supports evaluation, you can add these back:
    # evaluation_strategy="steps",
    # eval_steps=500,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Start training
trainer.train()
