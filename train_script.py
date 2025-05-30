# save as train_script.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

model_id = "esCyanide/ArcNemesis"
dataset_id = "0dAI/PentestingCommandLogic"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load dataset
dataset = load_dataset(dataset_id)

# Tokenize function
def tokenize_function(example):
    return tokenizer(example["input"] + "\n" + example["instruction"] + "\n" + example["output"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    push_to_hub=True,
    hub_model_id="esCyanide/ArcNemesis-finetuned",
    hub_strategy="every_save"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"] if "test" in tokenized_datasets else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
