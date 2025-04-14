#!/bin/bash

# Script to generate missing value dataset from PTB-XL

# Set paths
DATA_PATH="data/raw/ptb-xl-1.0.3"
OUTPUT_PATH="data/raw/ptb-xl-missing-values"

# Run the Python script
echo "Starting missing value dataset generation..."
python src/preprocess_missing_ecg.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Missing value dataset generation completed successfully!"
    echo "Output saved to: $OUTPUT_PATH"
else
    echo "Error: Missing value dataset generation failed!"
    exit 1
fi 