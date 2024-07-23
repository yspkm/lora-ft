import subprocess

MODELS = {
    "mistral_7b": {
        #"base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_model": "mistralai/Mistral-7B-v0.3",
        "lora_weights": {
            "math": "trained_models/mistral/math",
            "commonsense": "trained_models/mistral/commonsense",
        },
    },
    "llama_8b": {
        #"base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "lora_weights": {
            "math": "trained_models/llama/math",
            "commonsense": "trained_models/llama/commonsense",
        },
    },
    # "gemma_9b": {
    #     "base_model": "google/gemma-2-9b-it",
    #     "lora_weights": {
    #         "math": "trained_models/gemma/math",
    #         "commonsense": "trained_models/gemma/commonsense",
    #     },
    # },
}

datasets = {
    # "math": ["AQuA", "AddSub", "MultiArith", "SingleEq", "gsm8k", "SVAMP"],
    "math": ["SVAMP", "SingleEq", "AddSub", "MultiArith", "gsm8k"],
    #"commonsense": ["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Challenge","ARC-Easy","openbookqa",],
}

gpus = [0, 1, 2, 3]


def generate_commands():
    commands = []
    for model_name, model_info in MODELS.items():
        if 'gemma' in model_name:
            continue
        base_model = model_info["base_model"]
        for dataset_type, datasets_list in datasets.items():
            lora_weights = model_info["lora_weights"][dataset_type]
            for dataset in datasets_list:
                # fmt: off
                command = [
                    "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)),
                    "python3", f"evaluate_{dataset_type}.py",
                    "--model", model_name,
                    "--adapter", "LoRA",
                    "--dataset", dataset,
                    "--base_model", base_model,
                    "--lora_weights", lora_weights
                ]
                # fmt: on
                commands.append(command)
    return commands


def main():
    commands = generate_commands()
    for command in commands:
        print("Running command:", " ".join(command))
        subprocess.run(" ".join(command), shell=True)


if __name__ == "__main__":
    main()
