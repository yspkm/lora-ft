import os
import json
from transformers import pipeline, GenerationConfig
from datasets import load_dataset

import re
import fire
import sys
import argparse

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModelForCausalLM
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():

    args = parse_args()
    save_file = f"experiment/{args.model}-{args.adapter}-{args.dataset}.json"
    create_dir("experiment/")

    dataset = load_dataset(
        "json", data_files=f"dataset/{args.dataset}/test.json"
    )
    tokenizer, peft_model = load_model(args)

    text_generator = pipeline(
        task="text-generation", model=peft_model, tokenizer=tokenizer
    )

    miss = 0.001
    output_data = {"correct": 0, "total": 0, "accuracy": 0.0, "results": []}
    correct = 0
    total = 0

    with open(save_file, "w+") as f:

        def process_data(data):
            nonlocal correct
            nonlocal total
            instruction = data["instruction"]
            outputs = evaluate(
                instruction=instruction,
                text_generation_pipeline=text_generator,
                eos_token_id=tokenizer.eos_token_id,
            )
            label = data["answer"]
            flag = False
            if args.dataset.lower() in ["aqua"]:
                predict = extract_answer_letter(args, outputs)
                if label == predict:
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(args, outputs)
                if abs(label - predict) <= miss:
                    flag = True
            result = {
                "instruction": instruction,
                "output_pred": outputs,
                "pred": predict,
                "flag": flag,
                "answer": label,
            }
            output_data["results"].append(result)
            if flag:
                correct += 1
            total += 1
            output_data["total"] = total
            output_data["correct"] = correct
            output_data["accuracy"] = correct / total
            f.seek(0)
            f.truncate()
            json.dump(output_data, f, indent=4)
            f.flush()

        dataset.map(process_data)
        print(
            f"correct: {correct} / {total} \naccuracy: {round(100*correct/total, 2)}%"
        )
        f.seek(0)
        f.truncate()
        output_data["correct"] = correct
        output_data["accuracy"] = correct / total
        json.dump(output_data, f, indent=4)
    print("test finished")


def evaluate(
    instruction,
    text_generation_pipeline,
    eos_token_id=128001,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=5,
    max_new_tokens=256,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        num_beams=num_beams,
        **kwargs,
    )

    generated_texts = text_generation_pipeline(
        prompt,
        max_length=max_new_tokens,
        generation_config=generation_config,
        pad_token_id=eos_token_id,
        truncation=True,
    )

    output = generated_texts[0]["generated_text"]
    return (
        output.split("### Response:")[1].strip()
        if "### Response:" in output
        else output
    )


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP"],
        required=True,
    )
    parser.add_argument(
        "--model", choices=["LLaMA-7B", "BLOOM-7B", "GPT-j-6B"], required=True
    )
    parser.add_argument(
        "--adapter",
        choices=["LoRA", "AdapterP", "AdapterH", "Parallel", "Prefix"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--load_8bit", action="store_true", default=False)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(
            f"can not find base model name by the value: {args.model}"
        )
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(
            f"can not find lora weight, the value is: {lora_weights}"
        )

    load_8bit = args.load_8bit
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )  # fix zwq
        peft_model = PeftModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        peft_model = PeftModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        peft_model = PeftModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        peft_model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        peft_model.config.bos_token_id = 1
        peft_model.config.eos_token_id = 2

        if not load_8bit:
            peft_model.half()  # seems to fix bugs for some users.

        peft_model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            peft_model = torch.compile(peft_model)

    return tokenizer, peft_model


def load_instruction(args) -> str:
    instruction = ""
    if not instruction:
        raise ValueError("instruct not initialized")
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
        if not pred:
            return float("inf")
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(" not support dataset: {}".format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r"A|B|C|D|E", sentence_)
    if pred_answers:
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        return ""


if __name__ == "__main__":
    fire.Fire(main)
