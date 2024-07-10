from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess

MODELS = {
    "llama_8b": {
        "base_model": 'meta-llama/Meta-Llama-3-8B-Instruct',
        "lora_weights": 'trained_models/llama/math'
    },
    "gemma_9b": {
        "base_model": 'google/gemma-2-9b-it',
        "lora_weights": 'trained_models/gemma/math'
    },
    "phi_7b": {
        "base_model": 'microsoft/Phi-3-small-8k-instruct',
        "lora_weights": 'trained_models/phi/math'
    },
    "mistral_7b": {
        "base_model": 'mistralai/Mistral-7B-Instruct-v0.3',
        "lora_weights": 'trained_models/mistral/math'
    },
    "internlm_7b": {
        "base_model": 'internlm/internlm2_5-7b-chat',
        "lora_weights": 'trained_models/internlm/math'
    }
}

datasets = ['AQuA', 'AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'SVAMP']
gpus = [0, 1, 2, 3]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

def evaluate(model_name, dataset, gpu):
    model_info = MODELS[model_name]
    base_model = model_info["base_model"]
    lora_weights = model_info["lora_weights"]
    
    print(f'Evaluating model {model_name} on dataset {dataset} using GPU {gpu}')

    command = f"CUDA_VISIBLE_DEVICES={gpu} python3 evaluate_math.py \
               --model {model_name} \
               --adapter LoRA \
               --dataset {dataset} \
               --base_model {base_model} \
               --lora_weights {lora_weights}"

    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu

for gpu in gpus:
    gpu_queue.put(gpu)

for model_name in MODELS.keys():
    for dataset in datasets:
        tasks_queue.put((model_name, dataset))

num_processes = min(tasks_queue.qsize(), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, *tasks_queue.get(), gpu_queue.get()) for _ in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if not tasks_queue.empty():
            futures.append(executor.submit(evaluate, *tasks_queue.get(), gpu_queue.get()))
