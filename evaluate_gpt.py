import json
import requests
import os
import re
import yaml
import time
import logging
from typing import Dict, Any
from tqdm import tqdm
import fire

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_config_openai(config_file: str = "config.yaml") -> Dict[str, Any]:
    with open(config_file, "r") as f:
        return yaml.safe_load(f)["openai"]

def get_user_content(instruction: str, input: str = "") -> str:
    if len(input) == 0:
        return f"""
        ### Instruction:
        {instruction}

        ### Response:"""
    else:
        return f"""
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:"""

def get_system_content(config: Dict[str, Any], data_type: str) -> str:
    return config['system_content'][data_type]

def get_json_headers(config: Dict[str, Any]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}

def get_json_body(model: str, system_content: str, user_content: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }

def extract_answer_number(dataset: str, output: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        output = output.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", output)]
        if not pred:
            return float("inf")
        return float(pred[-1])
    else:
        raise NotImplementedError(f"Dataset not supported: {dataset}")

def extract_answer(dataset: str, output: str) -> str:
    output = output.strip().lower()
    if dataset == "boolq":
        pred_answers = re.findall(r"true|false", output)
    elif dataset == "AQuA":
        pred_answers = re.findall(r"solution1|solution2|solution3|solution4|solution5", output)
    elif dataset == "piqa":
        pred_answers = re.findall(r"solution1|solution2", output)
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", output)
    elif dataset == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", output)
    elif dataset == "winogrande":
        pred_answers = re.findall(r"option1|option2", output)
    else:
        pred_answers = []
    if not pred_answers:
        return ""
    return pred_answers[0]

def evaluate_single_item(config: Dict[str, Any], model_name: str, dataset_type: str, dataset: str, item: Dict[str, Any]) -> str:
    system_content = get_system_content(config, dataset_type)
    user_content = get_user_content(item["instruction"], item.get("input", ""))
    
    headers = get_json_headers(config)
    json_body = get_json_body(model_name, system_content, user_content)
    
    while True:
        response = requests.post(config['api_end_point'], headers=headers, json=json_body)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            retry_after = int(re.search(r'(\d+)ms', response.json()['error']['message']).group(1)) / 1000
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
        else:
            logging.error(f"Request failed with status code {response.status_code}. Retrying in 1 second.")
            time.sleep(1)

def evaluate_dataset(model_name: str, dataset_type: str, dataset: str):
    config = get_config_openai()
    miss = 0.001
    correct = 0
    total = 0
    output_data = {"correct": 0, "total": 0, "accuracy": 0.0, "results": []}

    save_file = f"experiment/{model_name}-0shot-{dataset}.json"
    os.makedirs("experiment", exist_ok=True)
    json_path = f"./dataset/{dataset}/test.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(save_file, "w+") as f:
        for item in tqdm(data, desc="Evaluating items"):
            output = evaluate_single_item(config, model_name, dataset_type, dataset, item)
            flag = False

            if dataset_type == "math":
                label: float = float(item.get('answer'))
                predict = extract_answer_number(dataset, output)
                if abs(label - predict) <= miss:
                    flag = True
            else:
                label: str = str(item.get("answer"))
                predict = extract_answer(dataset, output)
                if label == predict:
                    flag = True
            
            result = {
                "instruction": item.get("instruction"),
                "output_pred": output,
                "pred": predict,
                "flag": flag,
                "answer": label,
            }
            total += 1
            correct += 1 if flag else 0
            output_data["results"].append(result)
            output_data["total"] = total
            output_data["correct"] = correct
            output_data["accuracy"] = correct / total
            f.seek(0)
            f.truncate()
            json.dump(output_data, f, indent=4)
            f.flush()
        logging.info(f"Correct: {correct} / {total} \nAccuracy: {round(100*correct/total, 2)}%")

def main(model_name: str, dataset_type: str, dataset: str):
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"./logs/{model_name}-{dataset_type}-{dataset}.log"),  # Logs to 'app.log' file
                        logging.StreamHandler()          # Also logs to console
                    ])
    evaluate_dataset(model_name, dataset_type, dataset)

if __name__ == "__main__":
    fire.Fire(main)
