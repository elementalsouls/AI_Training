from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# -------------------------
# Config
# -------------------------
model_id = "esCyanide/ArcNemesis"
dataset_id = "0dAI/PentestingCommandLogic"
output_dir = "./arc_output"
max_length = 512
batch_size = 2

# -------------------------
# Load model & tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset(dataset_id)

# -------------------------
# Tokenization logic
# -------------------------
def tokenize_function(example):
    prompt = f"Instruction: {example['INSTRUCTION']}\nResponse: {example['RESPONSE']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# -------------------------
# Data collator
# -------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# Training args
# -------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation", tokenized_datasets["train"]),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------
# Train & save
# -------------------------
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
