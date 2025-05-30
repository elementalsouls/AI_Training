# -*- coding: utf-8 -*-
# This script fine-tunes the esCyanide/ArcNemesis model (loaded as a PEFT adapter)
# on the 0dAI/PentestingCommandLogic dataset from Hugging Face Hub.

# --- 1. Install Required Libraries ---
# These are the libraries needed for model loading, PEFT, training, and datasets.
# Using quiet install (-q) and upgrading (-U).

import subprocess
import sys
import time # Import time for output_dir timestamp

def install(package):
    print(f"Installing {package}...")
    try:
        # Ensure the install command is indented under the function definition
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Capture output/errors
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        # Decide if you want to exit on install failure
        sys.exit(f"Exiting due to installation failure for {package}.")
    except Exception as e:
        print(f"An unexpected error occurred during installation of {package}: {e}")
        sys.exit(f"Exiting due to unexpected installation error for {package}.")


# List of packages to install
required_packages = [
    "bitsandbytes",
    # --- UPDATED TRANSFORMERS DEPENDENCY ---
    # Removing the specific version constraint to get a version that supports Mistral.
    "transformers",
    # --- END UPDATED DEPENDENCY ---
    "peft",
    "accelerate",
    "datasets",
    "scipy", # Often needed by transformers/accelerate
    "einops", # Often needed by transformers
    # "evaluate", # Optional, for evaluation metrics
    # "trl",      # Optional, if using SFTTrainer instead of Trainer
    # "rouge_score", # Optional, for ROUGE metric (usually for summarization)
    "pandas", # Useful for data inspection
    "torch" # Ensure torch is installed and has CUDA support if needed
]

print("Starting required package installations...")
for package in required_packages:
    install(package)
print("Package installation complete.")

# Ensure torch is installed correctly with CUDA support if available
# This check should ideally be after installing 'torch'
try:
    import torch
    print(f"\nTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        # Check CUDA capability if using bfloat16
        # if torch.cuda.get_device_capability(0)[0] >= 8: # A100, H100 etc.
        #     print("GPU supports bfloat16.")
        #     # compute_dtype = getattr(torch, "bfloat16")
    else:
        print("CUDA is not available. Training will be on CPU (likely very slow).")
        # Decide if you want to exit if no GPU is found
        # sys.exit("Exiting: CUDA not available, GPU required for training.")
except ImportError:
    print("\nTorch is not installed or not found.")
    sys.exit("Exiting: Torch is required but not found.")
except Exception as e:
    print(f"\nAn error occurred while checking Torch/CUDA: {e}")
    sys.exit("Exiting due to Torch/CUDA check failure.")


# --- 2. Secure Hugging Face Hub Login ---
# This section handles logging into the Hugging Face Hub.
# DO NOT hardcode your token directly here. Use one of the secure methods below.

import os
from huggingface_hub import login, HfFolder

print("\nAttempting to log into Hugging Face Hub...")
try:
    # Attempt to log in. This will use the token from env var or CLI config if set.
    login()
    print("Successfully logged into Hugging Face Hub.")
    # Verify if a token file exists after login (optional)
    if HfFolder.get_token() is not None:
        print("Hugging Face token found (secured).")
    else:
        print("Hugging Face token not found after login attempt.")

except Exception as e:
    print(f"Hugging Face Hub login failed: {e}")
    print("Please ensure your Hugging Face token (with write permissions if pushing later)")
    print("is correctly set as an environment variable (HUGGING_FACE_HUB_TOKEN)")
    print("or you have logged in using 'huggingface-cli login' in your terminal.")
    # Decide if you want the script to exit here if login is crucial
    # sys.exit("Exiting due to Hugging Face Hub login failure.")


# Disable Weights and Biases if not needed (as in your notebook)
os.environ['WANDB_DISABLED'] = "true"
print("WANDB logging disabled.")


# --- 3. Define Model and Dataset IDs and Paths ---
# These are the IDs on Hugging Face Hub and paths for output

base_model_name = 'teknium/OpenHermes-2.5-Mistral-7B' # The base model ArcNemesis was likely built from
peft_model_id = 'esCyanide/ArcNemesis' # Your PEFT model ID on the Hub
huggingface_dataset_name = "0dAI/PentestingCommandLogic" # The dataset ID on the Hub
output_dir = f'./peft-pentesting-training-{str(int(time.time()))}' # Directory to save training artifacts
seed = 42 # Random seed for reproducibility

print(f"\nBase Model ID: {base_model_name}")
print(f"PEFT Model ID: {peft_model_id}")
print(f"Dataset ID: {huggingface_dataset_name}")
print(f"Output Directory: {output_dir}")
print(f"Random Seed: {seed}")


# --- 4. Load Base Model with Quantization Config ---
# We need the base model structure and its quantization details to load the PEFT adapter.

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Define BitsAndBytes config (assuming 4-bit QLoRA as in your notebook)
# Use bfloat16 if your GPU supports it (Ampere architecture or newer, e.g., A100, H100)
# Otherwise, float16 is standard.
compute_dtype = getattr(torch, "float16")
# if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
#      compute_dtype = getattr(torch, "bfloat16")
#      print("Using bfloat16 for compute_dtype.")
# else:
#      print("Using float16 for compute_dtype.")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Define device map
device_map = {"": 0} # Assuming training on a single GPU (device 0). Adjust if using multiple.

print(f"\nLoading base model: {base_model_name} with quantization config...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True, # Needed for some model architectures
        # use_auth_token=True, # Deprecated, login() handles auth
    )
    print("Base model loaded.")
except Exception as e:
    print(f"Error loading base model: {e}")
    sys.exit("Exiting due to base model loading failure.")


# --- 5. Load Tokenizer ---
# Load the tokenizer corresponding to the base model.

print(f"\nLoading tokenizer for: {base_model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        # padding_side="left", # Keep left padding for generation, but right padding is standard for training
        padding_side="right", # Set padding to right for training effectiveness
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False, # Use slow tokenizer if fast causes issues with special tokens
        # use_auth_token=True, # Deprecated
    )
    # Set pad_token to eos_token if it's not set
    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         print(f"Tokenizer pad_token set to eos_token ({tokenizer.eos_token_id}).")

    print("Tokenizer loaded.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit("Exiting due to tokenizer loading failure.")


# --- 6. Load the PEFT Adapter (Your ArcNemesis Model) ---
# Load the PEFT weights from your model ID on top of the base model.

from peft import PeftModel, prepare_model_for_kbit_training # LoRAConfig and get_peft_model imported earlier

print(f"\nLoading PEFT adapter: {peft_model_id}...")
try:
    # Prepare the base model for kbit training again before loading adapter
    # This ensures it's in the correct state for attaching the PEFT weights.
    # If your esCyanide/ArcNemesis model *is* the merged model, you would skip
    # loading base_model and prepare_model_for_kbit_training, and instead
    # load esCyanide/ArcNemesis directly using AutoModelForCausalLM.from_pretrained
    # with quantization config and then proceed to gradient_checkpointing_enable.
    # Assuming esCyanide/ArcNemesis is an adapter:
    base_model = prepare_model_for_kbit_training(base_model)
    print("Base model prepared for kbit training.")

    # Load the PEFT model by applying the adapter weights to the prepared base model
    # This assumes esCyanide/ArcNemesis contains only the PEFT adapter weights.
    model_to_train = PeftModel.from_pretrained(base_model, peft_model_id)
    print("PEFT adapter loaded onto base model.")
    # print(model_to_train) # Optional: print model structure

    # You might want to print trainable parameters to confirm PEFT is active
    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    print("\nTrainable parameters after loading PEFT adapter:")
    print(print_number_of_trainable_model_parameters(model_to_train))

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model_to_train.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled on model.")


except Exception as e:
    print(f"Error loading PEFT model: {e}")
    sys.exit("Exiting due to PEFT model loading failure.")


# --- 7. Load and Prepare Training Dataset ---
# Load the dataset from Hugging Face Hub and apply preprocessing.

from datasets import load_dataset, DatasetDict, Dataset
from functools import partial
# time already imported

print(f"\nLoading dataset: {huggingface_dataset_name}...")
try:
    # Load the dataset. It should be a DatasetDict.
    dataset = load_dataset(huggingface_dataset_name)
    print("Dataset loaded.")
    print(dataset)

    # Inspect dataset structure and column names
    # Assuming the '0dAI/PentestingCommandLogic' dataset has 'instruction' and 'response' columns
    # based on similar datasets. VERIFY THIS on the dataset page or by printing dataset['train'][0]
    dataset_input_col = 'instruction' # CHECK THE DATASET PAGE FOR ACTUAL COLUMN NAMES!
    dataset_output_col = 'response' # CHECK THE DATASET PAGE FOR ACTUAL COLUMN NAMES!

    if 'train' not in dataset:
         print("Error: Dataset does not have a 'train' split.")
         print("Available splits:", dataset.keys())
         sys.exit("Exiting due to missing train split in dataset.")

    if dataset_input_col not in dataset['train'].column_names or dataset_output_col not in dataset['train'].column_names:
        print(f"\nError: Expected columns '{dataset_input_col}' and '{dataset_output_col}' not found in dataset train split.")
        print("Available columns:", dataset['train'].column_names)
        sys.exit("Exiting due to dataset column mismatch.")

    print(f"\nDetected input column: '{dataset_input_col}'")
    print(f"Detected output column: '{dataset_output_col}'")
    print("\nFirst sample from train split:")
    print(dataset['train'][0])


except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit("Exiting due to dataset loading failure.")

# --- 7a. Define Data Preprocessing Functions ---
# Adapt these functions based on the actual column names and desired format of your dataset.

def create_prompt_formats(sample, input_col, output_col):
    """
    Format the sample based on the specified input and output columns
    into a prompt-response pair for causal language modeling.
    :param sample: Sample dictionary from the dataset.
    :param input_col: Name of the column containing the input text (instruction/query).
    :param output_col: Name of the column containing the target output text (response).
    """
    # This formatting should match how you want the model to see data during training
    # and how you will structure prompts during inference.
    # Example based on your previous notebook's format:
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct:" # Or similar key based on task
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    blurb = f"{INTRO_BLURB}\n\n" # Add newlines after blurb

    # Ensure columns exist in the sample dictionary and get their content
    input_text = sample.get(input_col, "").strip() # Use strip to remove leading/trailing whitespace
    output_text = sample.get(output_col, "").strip()

    if not input_text or not output_text:
        # Return a sample indicating it should be filtered out
        return {"text": None}

    # Combine parts. The model should learn to generate everything after RESPONSE_KEY
    # INCLUDING the RESPONSE_KEY itself if you want it to predict the format.
    # A common format is Instruction -> Response.
    # For instruction tuning, the format often includes special tokens/keys like below.

    # Format: Blurb \n\n Instruction_Key Input_Text \n\n Response_Key Output_Text \n\n End_Key
    # Ensure a space after keys for better tokenization boundary
    formatted_prompt = f"{blurb}{INSTRUCTION_KEY} {input_text}\n\n{RESPONSE_KEY} {output_text}\n\n{END_KEY}"

    sample["text"] = formatted_prompt # Add the formatted text as a new column

    return sample

def get_max_length(model):
    """Gets the maximum sequence length from the model configuration."""
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(conf, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024 # Default value if not found
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch of text samples.
    """
    # The DataCollatorForLanguageModeling will handle padding to the longest sequence in the batch
    # and creating labels. We just need to tokenize the text.
    return tokenizer(
        batch["text"], # Tokenize the 'text' column created by create_prompt_formats
        max_length=max_length,
        truncation=True,
        # padding='max_length' # Do not pad here when using DataCollatorForLanguageModeling
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset: DatasetDict, input_col: str, output_col: str):
    """
    Format & tokenize the dataset so it is ready for training.
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    :param seed (int): Random seed for shuffling
    :param dataset (DatasetDict): The input dataset
    :param input_col (str): Name of the column for input text.
    :param output_col (str): Name of the column for output text.
    """
    print("\nPreprocessing dataset...")

    # Apply create_prompt_formats to each sample to create the 'text' column
    print("Applying prompt formatting...")
    create_prompt_partial = partial(create_prompt_formats, input_col=input_col, output_col=output_col)
    # Note: map can process multiple splits if available in the DatasetDict
    dataset_with_text = dataset.map(create_prompt_partial)

    # Filter out samples where create_prompt_formats returned {"text": None}
    print("Filtering samples with empty text after formatting...")
    # Process each split in the DatasetDict
    for split in list(dataset_with_text.keys()):
         initial_num_rows = dataset_with_text[split].num_rows
         dataset_with_text[split] = dataset_with_text[split].filter(lambda sample: sample.get("text") is not None and len(sample["text"].strip()) > 0)
         filtered_num_rows = dataset_with_text[split].num_rows
         print(f"Filtered out {initial_num_rows - filtered_num_rows} samples with empty text after formatting from {split} split.")

    # If after filtering, any split becomes empty, handle that
    if len(dataset_with_text.keys()) == 0 or ('train' in dataset_with_text and dataset_with_text['train'].num_rows == 0):
        print("Error: No training samples remaining after formatting and filtering.")
        sys.exit("Exiting due to empty training dataset after preprocessing.")


    # Get the original column names before tokenization from the splits that still exist
    # We assume all splits have the same original columns
    sample_split_key = list(dataset_with_text.keys())[0] # Get the name of the first split
    original_columns_to_remove = [col for col in dataset_with_text[sample_split_key].column_names if col != 'text']
    # Ensure 'text' is also removed after tokenization
    all_columns_to_remove_after_tokenization = original_columns_to_remove + ['text']


    print(f"Original columns in dataset splits: {original_columns_to_remove}")
    print(f"Intermediate 'text' column added.")
    print(f"Columns to remove after tokenization map: {all_columns_to_remove_after_tokenization}")


    # Apply tokenization to each split
    print("Applying tokenization...")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    processed_dataset = dataset_with_text.map(
        _preprocessing_function,
        batched=True, # Process in batches for efficiency
        # Only remove columns that exist in the split currently being processed by map
        remove_columns=[col for col in all_columns_to_remove_after_tokenization if col in dataset_with_text[sample_split_key].column_names],
    )

    # Filter out samples that have input_ids exceeding max_length for each split
    print(f"Filtering samples exceeding max length ({max_length})...")
    for split in list(processed_dataset.keys()):
         if 'input_ids' in processed_dataset[split].column_names:
             initial_tokenized_rows = processed_dataset[split].num_rows
             processed_dataset[split] = processed_dataset[split].filter(lambda sample: len(sample["input_ids"]) <= max_length)
             filtered_tokenized_rows = processed_dataset[split].num_rows
             print(f"Filtered out {initial_tokenized_rows - filtered_tokenized_rows} samples exceeding max length from {split} split.")
         else:
             print(f"Warning: 'input_ids' column not found after tokenization in {split} split. Cannot filter by max_length.")

    # Shuffle the train split
    if 'train' in processed_dataset:
        print("Shuffling train split...")
        processed_dataset['train'] = processed_dataset['train'].shuffle(seed=seed)

    # Ensure the split names are correct for the Trainer ('train' and 'eval')
    # If the original dataset has 'validation' or 'test', rename it to 'eval'
    if 'validation' in processed_dataset and 'eval' not in processed_dataset:
        processed_dataset['eval'] = processed_dataset.pop('validation')
        print("Renamed 'validation' split to 'eval'.")
    elif 'test' in processed_dataset and 'eval' not in processed_dataset:
        processed_dataset['eval'] = processed_dataset.pop('test')
        print("Renamed 'test' split to 'eval'.")

    # Final check for required splits
    if 'train' not in processed_dataset:
        print("Error: 'train' split missing after preprocessing.")
        sys.exit("Exiting due to missing train split.")
    # 'eval' split is optional for the Trainer, but useful if available.


    print("\nProcessed DatasetDict:")
    print(processed_dataset)
    print("\nExample processed train sample:")
    if 'train' in processed_dataset and processed_dataset['train'].num_rows > 0:
        print(processed_dataset['train'][0])
    else:
        print("No samples in train split to display.")


    return processed_dataset

# --- 7b. Run Preprocessing and Split ---

# Get max length of the model (using model_to_train which is the PEFT model)
# Ensure model_to_train is defined from step 6.
# Handle case where model_to_train might not be defined if script failed earlier
if 'model_to_train' not in locals():
     print("Error: 'model_to_train' is not defined. Ensure model loading steps ran successfully.")
     sys.exit("Exiting.")

max_length = get_max_length(model_to_train)

# Preprocess the dataset loaded from Hugging Face Hub
# Pass the input and output column names identified in step 7
processed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset, dataset_input_col, dataset_output_col)

# Assign the train and eval splits to variables expected by the Trainer
# Assumes processed_dataset now has 'train' and potentially 'eval' keys
if 'train' in processed_dataset:
    train_dataset = processed_dataset['train']
else:
    print("Error: 'train' split not found in processed dataset after preprocessing.")
    sys.exit("Exiting: Missing train split.")

if 'eval' in processed_dataset:
    eval_dataset = processed_dataset['eval']
else:
     print("Warning: 'eval' split not found in processed dataset after preprocessing. Evaluation during training will be skipped.")
     eval_dataset = None # Set to None if no eval split is available


print("\nDatasets ready for training:")
print("train_dataset:", train_dataset)
print("eval_dataset:", eval_dataset)


# --- 8. Configure and Initialize the Trainer ---
# Setup training arguments and the Trainer object.

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
# set_seed already imported

# Set the random seed for reproducibility
# set_seed(seed) # Already called above after imports
# print(f"\nRandom seed set to {seed}.") # Already printed above

# Define your training arguments
# Review these parameters and adjust them based on your computational resources and desired training duration.
# You might need to adjust batch size and gradient accumulation based on your GPU memory.
peft_training_args = TrainingArguments(
    output_dir=output_dir,             # Directory to save checkpoints and logs
    warmup_steps=10,                   # Number of steps for linear warmup
    per_device_train_batch_size=1,     # Batch size per GPU/core (for training)
    gradient_accumulation_steps=4,     # Accumulate gradients over X batches
    max_steps=1000,                    # Total number of training steps (adjust as needed, 1000 is short for ~283k samples)
    # num_train_epochs=1,              # Alternatively, train for epochs (use max_steps OR num_train_epochs)
    learning_rate=2e-4,                # The initial learning rate
    optim="paged_adamw_8bit",          # Optimizer (optimized for 8-bit)
    logging_steps=50,                  # Log every X updates steps
    logging_dir="./logs",              # Directory for storing logs
    save_strategy="steps",             # Save checkpoints based on steps
    save_steps=100,                    # Save checkpoint every X steps
    # Evaluation strategy: set to "steps" to evaluate periodically, "epoch", or "no"
    evaluation_strategy="steps" if eval_dataset is not None else "no",
    eval_steps=100 if eval_dataset is not None else None, # Evaluate every X steps if eval_dataset exists
    do_eval=True if eval_dataset is not None else False,  # Whether to run evaluation
    gradient_checkpointing=True,       # Already enabled on the model, keep True here
    report_to="none",                  # Reporting backend (e.g., "none", "wandb")
    overwrite_output_dir=True,         # Overwrite the output directory
    group_by_length=True,              # Group samples by length for efficiency
    # More parameters to consider:
    # lr_scheduler_type="cosine",
    # weight_decay=0.001,
    # fp16=True, # Enable mixed precision (if using GPU)
    # logging_first_step=True,
    # save_total_limit=3,
    # load_best_model_at_end=True if eval_dataset is not None else False, # Requires load_best_model_at_end=True
    # metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    # greater_is_better=False if eval_dataset is not None else None,
    # dataloader_num_workers=0, # Adjust based on CPU cores and data loading speed
)

# Create the data collator
# Data collator for causal language modeling. It handles padding and creating labels.
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print("\nInitializing Trainer...")
# Initialize the Trainer
# Pass the tokenizer to the Trainer as well, although the data collator already has it,
# it can be useful for logging or specific trainer functionalities.
peft_trainer = Trainer(
    model=model_to_train,     # Your PEFT-enabled model
    train_dataset=train_dataset, # Your processed training dataset
    eval_dataset=eval_dataset,   # Your processed evaluation dataset (or None)
    args=peft_training_args,  # Your training arguments
    data_collator=data_collator, # The data collator
    tokenizer=tokenizer,       # Pass tokenizer to Trainer
)
print("Trainer initialized.")


# --- 9. Start Training ---
print("\nStarting training...")
try:
    peft_trainer.train()
    print("\nTraining finished successfully.")
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    # You might want to save the current state or log more details here
    # before exiting, depending on how you want to handle errors.
    sys.exit("Exiting due to training failure.")


# --- 10. Save the Final Model ---
# The Trainer saves checkpoints, but saving the final state explicitly is good practice.

print(f"\nSaving final model checkpoint to {output_dir}/final_checkpoint...")
try:
    # The Trainer's save_model method saves the PEFT adapter weights
    peft_trainer.save_model(f"{output_dir}/final_checkpoint")
    print("Final model checkpoint saved.")
except Exception as e:
    print(f"Error saving final model checkpoint: {e}")
    # Continue script execution or exit depending on how critical this step is
    # sys.exit("Exiting due to model saving failure.")


# --- 11. Push Model to Hugging Face Hub (Optional) ---
# This step requires your token to have write permissions.

print("\nAttempting to push the trained PEFT adapter to Hugging Face Hub...")
print(f"Pushing to repository: {peft_model_id}")

try:
    # Ensure the PEFT model is on the correct device if pushing directly after training
    # model_to_train.to('cpu') # Might be needed before pushing depending on setup
    # model_to_train.eval() # Set to evaluation mode if needed

    # The push_to_hub method on the PEFT model saves the adapter weights and pushes them.
    # Ensure you have write access to peft_model_id namespace.
    model_to_train.push_to_hub(peft_model_id) # This pushes only the adapter weights
    tokenizer.push_to_hub(peft_model_id) # Also push the tokenizer

    print(f"PEFT model adapter and tokenizer successfully pushed to https://huggingface.co/{peft_model_id}")

    # Optional: You might also want to save and push the merged model if you merged it.
    # This would typically happen after training.
    # For example, if you merge after training:
    # merged_model = model_to_train.merge_and_unload()
    # merged_model_repo_id = f"{peft_model_id}-merged" # Example repo name for merged model
    # merged_model.push_to_hub(merged_model_repo_id)
    # tokenizer.push_to_hub(merged_model_repo_id)


except Exception as e:
    print(f"Error pushing model to Hugging Face Hub: {e}")
    print("Please ensure you are logged in with a token that has WRITE permissions")
    print(f"for the repository '{peft_model_id}'.")
    # Do not exit here if pushing is optional

print("\nScript execution finished.")
