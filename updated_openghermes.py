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

# 1) Load the dataset
dataset = load_dataset("0dAI/PentestingCommandLogic")

# 2) Initialize tokenizer & model
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # FP16 for memory savings
)
model.config.use_cache = False  # required when fine‐tuning

# 3) Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)

# 4) (Remove any manual gradient_checkpointing_enable() call)
#    — we will not use gradient_checkpointing here, because it was
#      preventing gradients from flowing into the LoRA weights.

# 5) Print trainable parameters to confirm LoRA adapters show up
print("\n––– Trainable parameters (LoRA weights) –––")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  • {name}")

# 6) Tokenization function (returns plain Python lists, not PyTorch tensors)
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

    # Build “labels” so that pad_token_id → –100 (ignored in loss)
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
        for input_ids in tokenized["input_ids"]
    ]
    tokenized["labels"] = labels
    return tokenized

# 7) Tokenize the entire dataset
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 8) Sanity‐check one batch before training
sample_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=2,
    collate_fn=default_data_collator,
)
sample_batch = next(iter(sample_loader))

print("\n––– Sample batch structure –––")
print("Keys in batch:", sample_batch.keys())
print("  input_ids.shape:", sample_batch["input_ids"].shape)
print("  attention_mask.shape:", sample_batch["attention_mask"].shape)
print("  labels.shape:", sample_batch["labels"].shape)

# 9) Define TrainingArguments (NO gradient_checkpointing here)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=False  ← default is False, so we omit it explicitly
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# 10) Use default_data_collator (so “labels” pass through untouched)
data_collator = default_data_collator

# 11) Initialize Trainer (no label_names argument, since 4.52.4 doesn’t support it)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    data_collator=data_collator,
)

# 12) Launch training
trainer.train()
