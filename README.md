# LoRA Fine Tuning

This project focuses on fine-tuning and evaluating various large language models (LLMs) on commonsense reasoning and mathematical reasoning tasks. It utilizes Parameter-Efficient Fine-Tuning (PEFT), specifically LoRA, to adapt models like Llama, Gemma, and Mistral. The framework includes scripts for downloading models, fine-tuning, and evaluating them on a suite of standard benchmarks.

-----

## 1\. File Structure

The project is organized into the following directories and files:

```
.
├── DATA_LICENSE
├── LICENSE
├── README.md
├── config.yaml.template      # Template for configuration (e.g., API keys)
├── dataset/                  # Contains various datasets for evaluation
│   ├── AQuA/
│   ├── ARC-Challenge/
│   ├── ARC-Easy/
│   ├── AddSub/
│   ├── MultiArith/
│   ├── SVAMP/
│   ├── SingleEq/
│   ├── boolq/
│   ├── gsm8k/
│   ├── hellaswag/
│   ├── mathqa/
│   ├── mawps/
│   ├── openbookqa/
│   ├── piqa/
│   ├── social_i_qa/
│   └── winogrande/
├── download_models.py        # Script to download pre-trained models
├── evaluate_commonsense.py   # Script to evaluate models on commonsense datasets
├── evaluate_gpt.py           # Script to evaluate GPT models via API
├── evaluate_math.py          # Script to evaluate models on math datasets
├── experiment.sh             # Shell script to run evaluation experiments
├── finetune.py               # Script for fine-tuning models using PEFT
├── ft-training_set/          # Datasets for fine-tuning
│   ├── alpaca_data.json
│   ├── commonsense_170k.json
│   └── math_50k.json
├── init.sh                   # Initialization script to set up the environment
├── model_test.ipynb          # Jupyter notebook for interactive model testing
├── multi_dataset_eval.py     # Script for parallel evaluation on multiple datasets and GPUs
├── multi_dataset_eval_gpt.py # Script for batch evaluation of GPT models
├── multi_dataset_finetune.py # Script to automate fine-tuning across multiple models and datasets
└── peft/                     # Submodule for the PEFT library
```

-----

## 2\. Setup and Installation

To set up the project environment, follow these steps:

1.  **Clone the repository and create a virtual environment:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ./init.sh
    ```

2.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

3.  **Configure API Keys:**

    Rename `config.yaml.template` to `config.yaml` and add your API keys for services like Hugging Face and OpenAI.

    ```yaml
    huggingface:
      token: "YOUR_HUGGINGFACE_TOKEN"
    openai:
      api_key: "YOUR_OPENAI_API_KEY"
      api_end_point: "https://api.openai.com/v1/chat/completions"
      ## optional
      #system_content:
      #  math: "You are a math problem-solving assistant."
      #  commonsense: "You are a commonsense reasoning assistant."
    ```

-----

## 3\. Usage

### 3.1. Download Models

The `download_models.py` script downloads pre-trained models from Hugging Face. You can specify the models to download by editing the `models` dictionary in the script.

```python
# In download_models.py
models = {
    "BASE_MODEL_GEMMA_2B_IT": "google/gemma-2b-it",
    "BASE_MODEL_GEMMA_2B": "google/gemma-2b",
    # Add other models as needed
}
```

Run the script to download the models:

```bash
python3 download_models.py
```

### 3.2. Fine-Tuning Models

The `finetune.py` script fine-tunes a base model using a specified dataset. It leverages PEFT (LoRA) for efficient training.

**Example Usage:**

```bash
torchrun --nproc_per_node=4 --master_port=<your_master_port> finetune.py \
    --base_model 'google/gemma-2b-it' \
    --data_path 'ft-training_set/math_50k.json' \
    --output_dir './trained_models/gemma2b/math' \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.5 \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --wandb_project 'MyProject'
```

The `multi_dataset_finetune.py` script can be used to automate the fine-tuning process for multiple models and datasets.

### 3.3. Evaluating Models

There are separate evaluation scripts for different task types and models.

#### Evaluating Fine-Tuned Models

Use `evaluate_math.py` for mathematical reasoning tasks and `evaluate_commonsense.py` for commonsense reasoning tasks.

**Example for Math Evaluation:**

```bash
python3 evaluate_math.py \
    --model 'gemma_2b' \
    --adapter 'LoRA' \
    --dataset 'gsm8k' \
    --base_model 'google/gemma-2b-it' \
    --lora_weights './trained_models/gemma2b/math'
```

**Example for Commonsense Evaluation:**

```bash
python3 evaluate_commonsense.py \
    --model 'gemma_2b' \
    --adapter 'LoRA' \
    --dataset 'boolq' \
    --base_model 'google/gemma-2b-it' \
    --lora_weights './trained_models/gemma2b/commonsense'
```

To run evaluations in parallel across multiple datasets and GPUs, use the `multi_dataset_eval.py` script.

#### Evaluating GPT Models

The `evaluate_gpt.py` script evaluates models like GPT-3.5 and GPT-4 via the OpenAI API.

**Example Usage:**

```bash
python3 evaluate_gpt.py \
    --model_name 'gpt-3.5-turbo' \
    --dataset_type 'commonsense' \
    --dataset 'hellaswag'
```

The `multi_dataset_eval_gpt.py` script can be used for batch evaluations.

### 3.4. Interactive Model Testing

The `model_test.ipynb` notebook provides an interactive environment to load fine-tuned models and test them with custom prompts.

-----

## 4\. Scripts Overview

  - **`init.sh`**: Sets up the Python virtual environment and installs all required dependencies.
  - **`download_models.py`**: Downloads specified pre-trained models from the Hugging Face Hub.
  - **`finetune.py`**: Runs the fine-tuning process on a single model and dataset using PEFT.
  - **`evaluate_math.py`**: Evaluates a fine-tuned model on mathematical reasoning benchmark datasets.
  - **`evaluate_commonsense.py`**: Evaluates a fine-tuned model on commonsense reasoning benchmark datasets.
  - **`evaluate_gpt.py`**: Evaluates a GPT model (via API) on a specified dataset.
  - **`multi_dataset_finetune.py`**: A wrapper script to orchestrate the fine-tuning of multiple models on various datasets.
  - **`multi_dataset_eval.py`**: A script for running evaluations in parallel on multiple GPUs for different models and datasets.
  - **`multi_dataset_eval_gpt.py`**: A script to run evaluations for GPT models across multiple datasets.
  - **`experiment.sh`**: An example shell script to run a series of evaluations.
  - **`model_test.ipynb`**: A Jupyter notebook for interactively loading and testing models.
