import transformers
print(f"Transformers version: {transformers.__version__}")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load dataset from HF Hub
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Initialize tokenizer and model
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix for padding token error

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Use FP16 to save memory
)
model.config.use_cache = False  # Disable cache for training with gradient checkpointing

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
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for LoRA

# Ensure LoRA parameters require gradients
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

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
        return_tensors="pt",  # Return PyTorch tensors
        return_attention_mask=True
    )
    
    # Ensure input_ids and labels are tensors with requires_grad=False
    tokenized["input_ids"] = tokenized["input_ids"].to(torch.long)
    tokenized["attention_mask"] = tokenized["attention_mask"].to(torch.long)
    
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens for loss
    tokenized["labels"] = labels
    
    return tokenized

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Debug TrainingArguments source
print(f"TrainingArguments source: {TrainingArguments.__module__}")

# Training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Reduced for memory
    per_device_eval_batch_size=2,   # Reduced for memory
    gradient_accumulation_steps=4,  # Simulate larger batch size
    gradient_checkpointing=True,    # Save memory
    eval_strategy="no",
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,  # Enabled for GPU
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    data_collator=data_collator,
)

# Train!
trainer.train()
