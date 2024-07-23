#!/bin/bash

datasets=("AQuA" "boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")
models=("gpt-3.5-turbo")

evaluate() {
    model_name=$1
    dataset_type="commonsense"
    dataset=$2

    echo "Evaluating model: $model_name, dataset: $dataset"

    python3 evaluate_gpt.py --model_name "$model_name" --dataset_type "$dataset_type" --dataset "$dataset"

    if [ $? -eq 0 ]; then
        echo "Completed evaluation for model: $model_name, dataset: $dataset"
    else
        echo "Error in evaluating model: $model_name, dataset: $dataset"
    fi
}

for model_name in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        evaluate "$model_name" "$dataset"
    done
done
