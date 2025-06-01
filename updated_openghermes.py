import transformers
print(f"Transformers version: {transformers.__version__}")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Model name
model_name = "teknium/OpenHermes-2.5-Mistral-7B"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

# Debug: ensure LoRA layers are trainable
print("\nTrainable parameters after PEFT wrapping:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f" - {name}")

# Tokenization function
def tokenize_function(examples):
    prompts = [
        f"Instruction: {instr}\nOutput: {resp}"
        for instr, resp in zip(examples["INSTRUCTION"], examples["RESPONSE"])
    ]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )

    # Create labels with padding masked
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]
        for ids in tokenized["input_ids"]
    ]
    tokenized["labels"] = labels
    return tokenized

# Tokenize dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Inspect one batch
sample_loader = DataLoader(tokenized_datasets["train"], batch_size=2, collate_fn=default_data_collator)
sample_batch = next(iter(sample_loader))
print("\nSample batch keys:", sample_batch.keys())
print("input_ids shape:", sample_batch["input_ids"].shape)
print("labels shape:", sample_batch["labels"].shape)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Use simple data collator
data_collator = default_data_collator

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    data_collator=data_collator,
    label_names=["labels"],
)

# Start training
trainer.train()
