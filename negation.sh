#! /bin/bash

for model in \
    llama-3-8b \
    gemma-2-2b \
    gemma-2-9b \
    gpt2 \
    mistral-7b \
    gemma-2-2b-it \
    gemma-7b-it \
    mistral-7b-it
do
    python negation.py --model_name $model
done