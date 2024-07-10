from transformers import AutoModelForCausalLM, AutoTokenizer
import concurrent.futures

models = {
    "BASE_MODEL_LLAMA_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "BASE_MODEL_GEMMA_9B": "google/gemma-2-9b-it",
    "BASE_MODEL_MISTRAL_7B": "mistralai/Mistral-7B-Instruct-v0.3",
}

def download_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model_name, tokenizer, model

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(download_model, model_name): name for name, model_name in models.items()}
    for future in concurrent.futures.as_completed(futures):
        model_name, tokenizer, model = future.result()
        print(f"{model_name} download completed")

print("All models downloaded") 