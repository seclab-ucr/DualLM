#!/bin/bash

# Example usage of DualLM main.py
#python3 main.py --commits 9d7a0577c9db35c4cc52db90bc415ea248446472 --name test --repo-dir ../repos/linux --summary-file /tmp/summary.txt --data-dir ../data/


# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key-here"


# Example commits (these are example commit hashes, replace with real ones)
COMMITS="ff2047fb755d d1e7fd6462ca f4020438fab0 2c1f6951a8a8 79dc7e3f1cd3"
#COMMITS="9d7a0577c9db35c4cc52db90bc415ea248446472"
#COMMITS="79dc7e3f1cd3"
# Set up paths
REPO_DIR="./repos/linux"           # Path to Linux kernel repository
DATA_DIR="./data"                  # Path to store data
SUMMARY_FILE="./data/summary.txt"  # Path for summary output

# Create necessary directories
mkdir -p ./data/results

# Run the main script
python3 ./codes/step1.py \
    --commits $COMMITS \
    --name "test_dataset" \
    --repo-dir "$REPO_DIR" \
    --summary-file "$SUMMARY_FILE" \
    --out-file1 "./data/results/step1.txt" \
    --out-file2 "./data/results/step2.txt" \
    --data-dir "./data/"
