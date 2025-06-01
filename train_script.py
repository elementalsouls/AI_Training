import transformers
print(f"Transformers version: {transformers.__version__}")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load dataset from HF Hub
dataset = load_dataset("0dAI/PentestingCommandLogic")

# Initialize tokenizer and model
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use BitsAndBytesConfig for 8-bit quantization
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

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

# Tokenization function
def tokenize_function(examples):
    prompts = [
        f"Instruction: {instr}\nOutput: {resp}"
        for instr, resp in zip(examples["INSTRUCTION"], examples["RESPONSE"])
    ]
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    
    labels = tokenized["input_ids"].copy()
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
        for label_seq in labels
    ]
    tokenized["labels"] = labels
    return tokenized

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Debug TrainingArguments source
print(f"TrainingArguments source: {TrainingArguments.__module__}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="no",  # Changed from evaluation_strategy
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
)

# Train!
trainer.train()
