import transformers
print(f"Transformers version: {transformers.__version__}")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load dataset from Hugging Face Hub
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Initialize tokenizer and model
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Use FP16 for memory savings
)
model.config.use_cache = False  # Needed for gradient checkpointing

# Set up LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()  # Enable checkpointing for memory efficiency

# Tokenization function (returns regular Python lists, NOT tensors)
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
        return_attention_mask=True
    )
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
        for input_ids in tokenized["input_ids"]
    ]
    tokenized["labels"] = labels
    return tokenized

# Tokenize the dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Debug source of TrainingArguments
print(f"TrainingArguments source: {TrainingArguments.__module__}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_strategy="no",
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Use default data collator (does not modify already-prepared labels)
data_collator = default_data_collator

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    data_collator=data_collator,
)

# Start training
trainer.train()
