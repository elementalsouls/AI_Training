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
    load_in_8bit=True,      # quantized load
    device_map="auto"
)

# Set up LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust for your model's modules if needed
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with LoRA adapters for fine-tuning
model = get_peft_model(model, lora_config)

# Tokenization function for dataset (adjust column names as needed)
def tokenize_function(examples):
    # Build prompt with INSTRUCTION and RESPONSE from your dataset
    prompts = [
        f"Instruction: {instr}\nOutput: {resp}" 
        for instr, resp in zip(examples["INSTRUCTION"], examples["RESPONSE"])
    ]
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Prepare training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",  # or "steps"
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,  # Mixed precision if supported
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=False,
)

# Initialize Trainer with model, args, datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
)

# Start training
trainer.train()
