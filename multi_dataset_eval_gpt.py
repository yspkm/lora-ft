import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import queue
import os

datasets = {
    #"math": ["AddSub", "MultiArith", "SingleEq", "gsm8k", "SVAMP"],
    "commonsense": ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
}
# models = ["gpt-3.5-turbo", "gpt-4o"]
models = ["gpt-3.5-turbo"]
#models = ["gpt-4o"]
tasks_queue = queue.Queue()

for model_name in models:
    for dataset_type, dataset_list in datasets.items():
        for dataset in dataset_list:
            tasks_queue.put((model_name, dataset_type, dataset))

def evaluate(model_name, dataset_type, dataset):
    command = [
        "python3", "evaluate_gpt.py",
        "--model_name", model_name,
        "--dataset_type", dataset_type,
        "--dataset", dataset
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Completed evaluation for model: {model_name}, dataset: {dataset}")
        return True  
    except subprocess.CalledProcessError as e:
        print(f"Error in evaluating model: {model_name}, dataset: {dataset}")
        print(e.output)
        return False  

def main():
    num_processes = 1

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        while not tasks_queue.empty() or futures:
            while not tasks_queue.empty() and len(futures) < num_processes:
                task = tasks_queue.get()
                futures.append(executor.submit(evaluate, *task))

            completed_futures = [future for future in futures if future.done()]
            for future in completed_futures:
                futures.remove(future)
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
