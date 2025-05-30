from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load dataset directly from Hugging Face Hub
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Initialize tokenizer and model
model_name = "esCyanide/ArcNemesis"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization function
def tokenize_function(example):
    prompt = f"Instruction: {example['INSTRUCTION']}\nOutput: {example['RESPONSE']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

# Tokenize dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Use smaller subsets or full dataset depending on your setup
train_dataset = tokenized_datasets["train"]

# Define training arguments compatible with transformers v4.52.4
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    # 'evaluation_strategy' not supported in this version; omit or use 'eval_steps' if needed
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
