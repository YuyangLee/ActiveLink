#!/bin/bash

# Create an output directory for logs if it doesn't exist
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# Define models to train and test
MODELS=("TransE" "TransR" "ComplEx" "RotatE")

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Starting training and testing for model: $MODEL"
    for DATASET_DIR in ./data/data_split*/; do
        DATASET_NAME=$(basename "$DATASET_DIR")
        echo "Running on dataset $DATASET_NAME"
        python3 train.py --model "$MODEL" --data_dir "$DATASET_DIR" | tee -a "$LOG_DIR/${MODEL}_output.log"
    done
    echo "Output saved to $LOG_DIR/${MODEL}_output.log"
done
