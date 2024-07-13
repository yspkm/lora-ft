#!/bin/bash

python3 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

git clone https://github.com/yspkm/peft.git
pip3 install --upgrade pip wheel setuptools 
pip3 install torch --index-url https://download.pytorch.org/whl/cu121 
pip3 install transformers datasets accelerate sentencepiece tiktoken
pip3 install fire gradio bitsandbytes appdirs black black[jupyter] einops matplotlib tqdm torchinfo tensorboard wandb pandas numpy pytest
pip3 install ipywidgets ipykernel jupyterlab