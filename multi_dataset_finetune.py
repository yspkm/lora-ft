import os
import subprocess
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "gemma_9b": "google/gemma-2-9b-it",
}

DATASETS = {
    "math": "ft-training_set/math_50k.json",
    "commonsense": "ft-training_set/commonsense_170k.json",
}

OUTPUT_DIRS = {
    "llama_8b_math": "trained_models/llama/math",
    "llama_8b_commonsense": "trained_models/llama/commonsense",
    "gemma_9b_math": "trained_models/gemma/math",
    "gemma_9b_commonsense": "trained_models/gemma/commonsense",
    "mistral_7b_math": "trained_models/mistral/math",
    "mistral_7b_commonsense": "trained_models/mistral/commonsense",
}

TRAINING_CONFIG = {
    "batch_size": 128,
    "micro_batch_size": 16,
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "cutoff_len": 256,
    "val_set_size": 2048,
    "adapter_name": "lora",
    "lr_scheduler_type": "cosine",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.5,
    "lora_target_modules": '["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
    "wandb_project": "inception_lora",
    "wandb_watch": "all",
    "wandb_log_model": "false",
    # wandb.watch메트릭 업로드 시간 때문에 싱클르 위해 10에서 20으로 늘림
    "logging_steps": 20,
    # Trainer에서 logging시에 eval때는 따로 루틴이 생성되는 것 같음
    # training metric이 로깅이 안 되므로 배수 관계가 아닌 값을 사용해야 함.
    "eval_step": 201,
    "save_step": 201,
}
CUR_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)


def train_model(model_name, base_model, data_path, output_dir):
    dataset_type = "math" if "math" in output_dir else "commonsense"
    wandb_run_name = (
        f"{model_name}_{TRAINING_CONFIG['adapter_name']}_{dataset_type}_{CUR_DATETIME}"
    )
    # fmt: off
    command = [
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "python3",
        "finetune.py",
        "--lora_r", str(TRAINING_CONFIG["lora_r"]),
        "--lora_alpha", str(TRAINING_CONFIG["lora_alpha"]),
        "--lora_dropout", str(TRAINING_CONFIG["lora_dropout"]),
        "--adapter_name", TRAINING_CONFIG["adapter_name"],
        "--lora_target_modules", TRAINING_CONFIG["lora_target_modules"],
        "--lr_scheduler_type", str(TRAINING_CONFIG["lr_scheduler_type"]),
        "--base_model", base_model,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--batch_size", str(TRAINING_CONFIG["batch_size"]),
        "--micro_batch_size", str(int(TRAINING_CONFIG["micro_batch_size"] / (2 if "9b" in model_name else 1))),
        "--num_epochs", str(TRAINING_CONFIG["num_epochs"]),
        "--learning_rate", str(TRAINING_CONFIG["learning_rate"]),
        "--cutoff_len", str(TRAINING_CONFIG["cutoff_len"]),
        "--val_set_size", str(TRAINING_CONFIG["val_set_size"]),
        "--wandb_project", TRAINING_CONFIG["wandb_project"],
        "--wandb_run_name", wandb_run_name,
        "--wandb_watch", TRAINING_CONFIG["wandb_watch"],
        "--wandb_log_model", TRAINING_CONFIG["wandb_log_model"],
        "--logging_steps", str(TRAINING_CONFIG["logging_steps"]),
        "--eval_step", str(TRAINING_CONFIG["eval_step"]),
        "--save_step", str(TRAINING_CONFIG["save_step"]),
    ]
    # fmt: on
    command_str = " ".join(command)
    print(f"Training {model_name} with dataset {data_path}")
    print(f"Running command: {command_str}")

    subprocess.run(command_str, shell=True, check=True)


for task_key, data_path in DATASETS.items():
    for model_key, base_model in MODELS.items():
        output_dir_key = f"{model_key}_{task_key}"
        output_dir = OUTPUT_DIRS[output_dir_key]
        train_model(model_key, base_model, data_path, output_dir)
