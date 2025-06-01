from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Verify CUDA availability
if not torch.cuda.is_available():
    print("Error: CUDA is not available. Please install a CUDA-compatible PyTorch version.")
    exit(1)

# Configuration
model_name = "/workspace/fine-tuning/arc_finetune/models/base"
dataset_path = "/workspace/fine-tuning/arc_finetune/dataset/train.jsonl"
output_dir = "/workspace/fine-tuning/arc_finetune/models/checkpoints"

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to("cuda")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

# Load dataset
try:
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if "INSTRUCTION" not in dataset.column_names or "RESPONSE" not in dataset.column_names:
        raise ValueError("Dataset must contain 'INSTRUCTION' and 'RESPONSE' columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Tokenization with dynamic padding and chat template
def tokenize_function(examples):
    prompts = []
    for instr, resp in zip(examples["INSTRUCTION"], examples["RESPONSE"]):
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": instr},
                {"role": "assistant", "content": resp}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            prompt = f"Instruction: {instr}\nOutput: {resp}"
        prompts.append(prompt)
    tokenized = tokenizer(prompts, padding=True, truncation=True, max_length=512)
    tokenized["labels"] = [[(token if token != tokenizer.pad_token_id else -100) for token in seq] for seq in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=4)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="/workspace/fine-tuning/arc_finetune/logs",
    save_strategy="steps",
    save_total_limit=2,
    evaluation_strategy="no",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_first_step=True,
    report_to=["tensorboard"]
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()

# Save fine-tuned LoRA model
model.save_pretrained(output_dir + "/final")
tokenizer.save_pretrained(output_dir + "/final")
print("Training complete")
