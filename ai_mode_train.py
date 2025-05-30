# -*- coding: utf-8 -*-
# This script fine-tunes the esCyanide/ArcNemesis model (loaded as a PEFT adapter)
# on the 0dAI/PentestingCommandLogic dataset from Hugging Face Hub.

# --- INSTRUCTIONS BEFORE RUNNING ---
# 1. Determine your system's CUDA version (e.g., by running `nvcc --version` in your terminal).
#    If you don't have `nvcc`, you might need to install the NVIDIA CUDA Toolkit or check
#    `nvidia-smi` output for the CUDA version supported by your driver.
# 2. Go to the official PyTorch installation page (https://pytorch.org/get-started/locally/).
# 3. Select your OS (Linux), Package (Pip), Language (Python), and your CUDA version
#    (match the version found in step 1).
# 4. Copy the EXACT installation command provided (it will include `--index-url https://download.pytorch.org/whl/cuXXX`, where XXX is your CUDA version).
# 5. Run that PyTorch installation command in your terminal BEFORE running this script.
#    Example for CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#    Example for CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 6. Ensure your Hugging Face write token (needed for pushing the model later) is set
#    as an environment variable (`export HUGGING_FACE_HUB_TOKEN="hf_YOUR_WRITE_TOKEN_HERE"`)
#    or you have logged in using `huggingface-cli login` in your terminal.
# -----------------------------------


# --- 1. Install Required Libraries ---
# These are the libraries needed for model loading, PEFT, training, and datasets.
# Using quiet install (-q) and upgrading (-U).
# TORCH IS NOT INSTALLED HERE - install it separately as per instructions above.

import subprocess
import sys
import time # Import time for output_dir timestamp

def install(package):
    print(f"Installing {package}...")
    try:
        # Ensure the install command is indented under the function definition
        # Using --no-deps for bitsandbytes might help prevent it from pulling a torch version,
        # but --no-deps can sometimes cause other issues. Let's try without for now.
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


# List of packages to install (excluding torch)
required_packages = [
    "bitsandbytes",
    "transformers", # Will install the latest version compatible with other deps (should support Mistral)
    "peft",
    "accelerate",
    "datasets",
    "scipy", # Often needed by transformers/accelerate
    "einops", # Often needed by transformers
    # "evaluate", # Optional, for evaluation metrics
    # "trl",      # Optional, if using SFTTrainer instead of Trainer
    # "rouge_score", # Optional, for ROUGE metric (usually for summarization)
    "pandas", # Useful for data inspection
    # --- TORCH IS REMOVED from this list ---
    # It MUST be installed manually with the correct CUDA version before running this script.
    # --- END REMOVED ---
]

print("Starting required package installations (excluding torch)...")
for package in required_packages:
    install(package)
print("Package installation complete.")

# Ensure torch is available and has CUDA support (CHECK AFTER installing torch manually)
# This check needs torch to be available in the environment *before* running this script.
try:
    import torch
    print(f"\nTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print("Torch appears to be installed correctly with CUDA support.")
        # Check CUDA capability if using bfloat16 (Optional, based on your needs)
        # if torch.cuda.get_device_capability(0)[0] >= 8: # Ampere or newer
        #      print("GPU supports bfloat16 compute capability.")
        #      # compute_dtype = getattr(torch, "bfloat16") # You might set this later
    else:
        print("CUDA is not available. Training will be on CPU (likely very slow).")
        print("Please ensure you installed the CUDA version of PyTorch correctly from pytorch.org.")
        # Decide if you want to exit if no GPU is found - highly recommended for LLM training
        sys.exit("Exiting: CUDA not available, GPU required for large model training.") # Added exit if no GPU
except ImportError:
    print("\nError: Torch is not installed or not found.")
    print("Please install PyTorch with CUDA support manually using the command from pytorch.org BEFORE running this script.")
    sys.exit("Exiting: Torch is required but not found.")
except Exception as e:
    print(f"\nAn error occurred while checking Torch/CUDA: {e}")
    sys.exit("Exiting due to Torch/CUDA check failure.")


# --- 2. Secure Hugging Face Hub Login ---
# This section handles logging into the Hugging Face Hub.
# DO NOT hardcode your token directly here. Use one of the secure methods mentioned in instructions.

import os
from huggingface_hub import login, HfFolder

print("\nAttempting to log into Hugging Face Hub...")
try:
    # Attempt to log in. This will use the token from env var or CLI config if set.
    # Added add_to_git_credential=False to prevent the interactive prompt in non-interactive scripts
    login(add_to_git_credential=False)
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
import torch # Imported and checked in Section 1

# Define BitsAndBytes config (assuming 4-bit QLoRA as in your notebook)
# Use bfloat16 if your GPU supports it (Ampere architecture or newer, e.g., A100, H100)
# Otherwise, float16 is standard. Check your GPU capability if unsure.
compute_dtype = getattr(torch, "float16")
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
     compute_dtype = getattr(torch, "bfloat16")
     print("Using bfloat16 for compute_dtype.")
else:
     print("Using float16 for compute_dtype.") # Default

print(f"Using compute_dtype: {compute_dtype}")

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

# AutoTokenizer is already imported in Section 4
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
    # This is done *after* loading the PEFT adapter
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

    # Check if necessary columns exist in the training split
    if dataset_input_col not in dataset['train'].column_names or dataset_output_col not in dataset['train'].column_names:
        print(f"\nError: Expected columns '{dataset_input_col}' and '{dataset_output_col}' not found in dataset train split.")
        print("Available columns:", dataset['train'].column_names)
        sys.exit("Exiting due to dataset column mismatch.")

    print(f"\nDetected input column: '{dataset_input_col}'")
    print(f"Detected output column: '{dataset_output_col}'")
    print("\nFirst sample from train split:")
    if dataset['train'].num_rows > 0:
         print(dataset['train'][0])
    else:
         print("Train split is empty.")


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
    # Use .get with a default empty string to handle missing keys gracefully
    input_text = sample.get(input_col, "").strip() # Use strip to remove leading/trailing whitespace
    output_text = sample.get(output_col, "").strip()

    if not input_text or not output_text:
        # Return a sample indicating it should be filtered out later
        return {"text": None} # Use None to indicate this sample should be dropped

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
    :returns: A DatasetDict with processed splits ('train' and potentially 'eval').
    """
    print("\nPreprocessing dataset...")

    # Apply create_prompt_formats to each sample to create the 'text' column
    print("Applying prompt formatting...")
    create_prompt_partial = partial(create_prompt_formats, input_col=input_col, output_col=output_col)
    # Note: map can process multiple splits if available in the DatasetDict
    dataset_with_text = dataset.map(create_prompt_partial)

    # Filter out samples where create_prompt_formats returned {"text": None} or empty text
    print("Filtering samples with empty text after formatting...")
    splits_to_process = list(dataset_with_text.keys())
    for split in splits_to_process:
         initial_num_rows = dataset_with_text[split].num_rows
         # Filter keeps samples where the lambda function returns True
         dataset_with_text[split] = dataset_with_text[split].filter(lambda sample: sample.get("text") is not None and len(sample["text"].strip()) > 0)
         filtered_num_rows = dataset_with_text[split].num_rows
         print(f"Filtered out {initial_num_rows - filtered_num_rows} samples with empty text after formatting from {split} split.")

    # If after filtering, any split becomes empty, handle that
    if 'train' not in dataset_with_text or dataset_with_text['train'].num_rows == 0:
        print("Error: No training samples remaining after formatting and filtering.")
        sys.exit("Exiting due to empty training dataset after preprocessing.")


    # Get the original column names before tokenization from the splits that still exist
    # We assume all splits have the same original columns for removal purposes
    # Take column names from the training split (assuming train split exists and has samples)
    sample_split_key = 'train' # Assumes train split exists and is not empty
    original_columns_to_remove = [col for col in dataset_with_text[sample_split_key].column_names if col != 'text']
    # Ensure 'text' is also removed after tokenization
    all_columns_to_remove_after_tokenization = original_columns_to_remove + ['text']


    print(f"Original columns in dataset splits: {original_columns_to_remove}")
    print(f"Intermediate 'text' column added.")
    print(f"Columns to remove after tokenization map: {all_columns_to_remove_after_tokenization}")


    # Apply tokenization to each split
    print("Applying tokenization...")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    # map applies the function to each split. We need to specify the columns to remove
    # which should be consistent across splits containing those columns.
    # Let's use the columns determined from the 'train' split.
    cols_to_remove = [col for col in dataset_with_text['train'].column_names if col != 'text'] + ['text'] # Get cols from train split

    processed_dataset = dataset_with_text.map(
        _preprocessing_function,
        batched=True,
        # Remove columns that exist in the *current* split being processed by map.
        # This requires re-filtering the list based on the split's columns if splits differ.
        # However, typically original columns are same across splits. Let's simplify:
        # remove_columns=cols_to_remove # This works if all splits have these columns
        # A more robust way if splits might have different original cols (less common):
        remove_columns = [col for col in dataset_with_text['train'].column_names if col != 'text'] + ['text'] # Assuming train has all relevant original cols


    )

    # Filter out samples that have input_ids exceeding max_length for each split
    print(f"Filtering samples exceeding max length ({max_length})...")
    splits_after_tokenize = list(processed_dataset.keys())
    for split in splits_after_tokenize:
         if 'input_ids' in processed_dataset[split].column_names:
             initial_tokenized_rows = processed_dataset[split].num_rows
             processed_dataset[split] = processed_dataset[split].filter(lambda sample: len(sample["input_ids"]) <= max_length)
             filtered_tokenized_rows = processed_dataset[split].num_rows
             print(f"Filtered out {initial_tokenized_rows - filtered_tokenized_rows} samples exceeding max length from {split} split.")
             # If a split becomes empty after filtering, remove it from the DatasetDict
             if processed_dataset[split].num_rows == 0:
                  print(f"Split '{split}' is empty after filtering and will be removed.")
                  processed_dataset.pop(split)

         else:
             print(f"Warning: 'input_ids' column not found after tokenization in {split} split.")
             # Decide how to handle splits without input_ids - maybe remove them too?
             print(f"Split '{split}' missing 'input_ids' and will be removed.")
             processed_dataset.pop(split)


    # Ensure the split names are correct for the Trainer ('train' and 'eval')
    # If the original dataset has 'validation' or 'test', rename it to 'eval'
    if 'validation' in processed_dataset and 'eval' not in processed_dataset:
        processed_dataset['eval'] = processed_dataset.pop('validation')
        print("Renamed 'validation' split to 'eval'.")
    elif 'test' in processed_dataset and 'eval' not in processed_dataset:
        processed_dataset['eval'] = processed_dataset.pop('test')
        print("Renamed 'test' split to 'eval'.")

    # Final check for required splits
    if 'train' not in processed_dataset or processed_dataset['train'].num_rows == 0:
        print("Error: 'train' split missing or empty after preprocessing.")
        sys.exit("Exiting due to missing or empty train split.")
    # 'eval' split is optional for the Trainer, but useful if available.


    print("\nProcessed DatasetDict:")
    print(processed_dataset)
    print("\nExample processed train sample (decoded):")
    if 'train' in processed_dataset and processed_dataset['train'].num_rows > 0:
        # Decode a sample to verify formatting after tokenization
        decoded_text = tokenizer.decode(processed_dataset['train'][0]['input_ids'], skip_special_tokens=False)
        print(decoded_text)
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
# Ensure dataset, dataset_input_col, and dataset_output_col are defined from step 7
if 'dataset' not in locals():
    print("Error: Dataset variable not found. Ensure dataset loading in step 7 ran.")
    sys.exit("Exiting.")
if 'dataset_input_col' not in locals() or 'dataset_output_col' not in locals():
    print("Error: Dataset column names not identified. Ensure step 7 ran successfully.")
    sys.exit("Exiting.")


processed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset, dataset_input_col, dataset_output_col)

# Assign the train and eval splits to variables expected by the Trainer
# Assumes processed_dataset now has 'train' and potentially 'eval' keys
if 'train' in processed_dataset:
    train_dataset = processed_dataset['train']
    if train_dataset.num_rows == 0:
        print("Error: train_dataset is empty after preprocessing.")
        sys.exit("Exiting: train_dataset is empty.")
else:
    print("Error: 'train' split not found in processed dataset after preprocessing.")
    sys.exit("Exiting: Missing train split.")

if 'eval' in processed_dataset:
    eval_dataset = processed_dataset['eval']
    if eval_dataset.num_rows == 0:
         print("Warning: eval_dataset is empty after preprocessing. Evaluation will be skipped.")
         eval_dataset = None # Set to None if empty
else:
     print("Warning: 'eval' split not found in processed dataset after preprocessing. Evaluation during training will be skipped.")
     eval_dataset = None # Set to None if missing


print("\nDatasets ready for training:")
