import os
import subprocess
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODELS = {
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma_9b": "google/gemma-2-9b-it",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3"
}

DATASETS = {
    "math": "ft-training_set/math_50k.json",
    "commonsense": "ft-training_set/commonsense_170k.json"
}

OUTPUT_DIRS = {
    "llama_8b_math": "trained_models/llama/math",
    "llama_8b_commonsense": "trained_models/llama/commonsense",
    "gemma_9b_math": "trained_models/gemma/math",
    "gemma_9b_commonsense": "trained_models/gemma/commonsense",
    "mistral_7b_math": "trained_models/mistral/math",
    "mistral_7b_commonsense": "trained_models/mistral/commonsense"
}

TRAINING_CONFIG = {
    "batch_size": 128,
    "micro_batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 3e-4,
    "cutoff_len": 256,
    "val_set_size": 120,
    "adapter_name": "lora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.5,
    "wandb_project": "inception_lora",
    "wandb_watch": "all",
    "wandb_log_model": "true",
    "lr_scheduler_type": "cosine",
    "target_modules": '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
}

for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

def train_model(model_name, base_model, data_path, output_dir):
    cur_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_type = "math" if "math" in output_dir else "commonsense"
    wandb_run_name = f"{model_name}_{dataset_type}_{cur_datetime}"
    
    command = [
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "python3",
        "finetune.py",
        "--lora_r", str(TRAINING_CONFIG["lora_r"]),
        "--lora_alpha", str(TRAINING_CONFIG["lora_alpha"]),
        "--lora_dropout", str(TRAINING_CONFIG["lora_dropout"]),
        "--lr_scheduler_type", str(TRAINING_CONFIG["lr_scheduler_type"]),
        "--base_model", base_model,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--batch_size", str(TRAINING_CONFIG["batch_size"]),
        "--micro_batch_size", str(TRAINING_CONFIG["micro_batch_size"]),
        "--num_epochs", str(TRAINING_CONFIG["num_epochs"]),
        "--learning_rate", str(TRAINING_CONFIG["learning_rate"]),
        "--cutoff_len", str(TRAINING_CONFIG["cutoff_len"]),
        "--val_set_size", str(TRAINING_CONFIG["val_set_size"]),
        "--adapter_name", TRAINING_CONFIG["adapter_name"],
        "--wandb_project", TRAINING_CONFIG["wandb_project"],
        "--wandb_run_name", wandb_run_name,
        "--wandb_watch", TRAINING_CONFIG["wandb_watch"],
        "--wandb_log_model", TRAINING_CONFIG["wandb_log_model"],
        "--target_modules", TRAINING_CONFIG["target_modules"]
    ]

    command_str = " ".join(command)
    print(f"Training {model_name} with dataset {data_path}")
    print(f"Running command: {command_str}")
    
    subprocess.run(command_str, shell=True, check=True)

for model_key, base_model in MODELS.items():
    for task_key, data_path in DATASETS.items():
        output_dir_key = f"{model_key}_{task_key}"
        output_dir = OUTPUT_DIRS[output_dir_key]
        train_model(model_key, base_model, data_path, output_dir)