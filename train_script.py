from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load dataset from HF Hub (replace with your dataset if local)
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Initialize tokenizer and model (load quantized 8-bit model)
model_name = "esCyanide/ArcNemesis"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,      # requires bitsandbytes installed
    device_map="auto"
)

# Set up LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # adjust if your model's architecture differs
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with LoRA adapters for fine-tuning
model = get_peft_model(model, lora_config)

# Tokenization function with padding and label masking (-100 for pad tokens)
def tokenize_function(examples):
    prompts = [
        f"Instruction: {instr}\nOutput: {resp}"
        for instr, resp in zip(examples["INSTRUCTION"], examples["RESPONSE"])
    ]
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    
    labels = tokenized["input_ids"].copy()
    # Replace padding token ids by -100 so they are ignored by the loss function
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
        for label_seq in labels
    ]
    tokenized["labels"] = labels
    return tokenized

# Tokenize the dataset (remove original columns to keep only tokenized inputs)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",   # set "steps" if you want eval during training
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,                  # use mixed precision if your GPU supports it
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),  # may be None if no validation split
    tokenizer=tokenizer,
)

# Train!
trainer.train()
