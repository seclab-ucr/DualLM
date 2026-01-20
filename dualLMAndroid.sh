#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate dl_step1
source .env
# Set up environment variables
# Example commits (these are example commit hashes, replace with real ones)
#linux kernel cve
#COMMITS="56b88b50565cd8b946a2d00b0c83927b7ebb055e 1ee9d9122801eb688783acd07791f2906b87cb4f 84175dc5b2c932266a50c04e5ce342c30f817a2f 27c5a095e2518975e20a10102908ae8231699879 a30f895ad3239f45012e860d4f94c1a388b36d14 ff2047fb755d d1e7fd6462ca f4020438fab0 2c1f6951a8a8 79dc7e3f1cd3"
COMMITS="bce004fba8be9e1bb575301f398b3ecc27ba42de" #andorid cve

echo "Cleaning up previous results..."
rm -rf ./data/results/*
rm -rf ./data/train_valid_test/*
rm -rf ./data/summary.txt
rm -rf ./data/raw_data/*
rm -rf ./data/slicing_diff/*
rm -rf ./slices/*

if [ ! -d "./repos/linux_backup" ]; then
    # If linux_backup doesn't exist, perform the moves
    mv ./repos/linux ./repos/linux_backup
    mv ./repos/android ./repos/linux
    #echo "Moved repositories successfully"
else
    #echo "./repos/linux_backup already exists, skipping move operation"
    echo ""
fi

echo "******************* Android Kenel Commits to analyze *******************"
echo "********************************************************************************"
echo "$COMMITS"
echo "********************************************************************************"


# Set up paths
DATASET_NAME="test_dataset"
REPO_DIR="./repos/linux"           # Path to Linux kernel repository
DATA_DIR="./data"                  # Path to store data
SUMMARY_FILE="./data/summary.txt"  # Path for summary output

# Create necessary directories
mkdir -p ./data/results

echo "********************************************************************************"
echo "Step 1: Analyzing commits with ChatGPT"
echo "********************************************************************************"
# Run the main script
python3 ./codes/step1.py \
    --commits $COMMITS \
    --name "$DATA_SET_NAME" \
    --repo-dir "$REPO_DIR" \
    --summary-file "$SUMMARY_FILE" \
    --out-file1 "./data/results/step1.txt" \
    --out-file2 "./data/results/step2.txt" \
    --data-dir "./data/" \
    --not-reliable "./data/results/not_reliable.txt"

conda deactivate

echo "********************************************************************************"
echo "Step 2: Analyzing commits that cannot be clasffified by ChatGPT with slicing data and customzied models."
echo "********************************************************************************"

conda activate dl_step2

cd codes
./eval_given_step1.sh 0
./eval_given_step2.sh 0

conda deactivate

conda activate dl_step1

cd ../

echo "********************************************************************************"
echo "Step 3: Final step: Output the results"
echo "********************************************************************************"

NOT_RELIABLE_COMMITS=$(paste -sd' ' ./data/results/not_reliable.txt)

#echo "Not reliable commits (single line):"
#echo "$NOT_RELIABLE_COMMITS"
python3 ./codes/final.py \
    --step1-out-file "./data/results/step1_res.txt" \
    --step2-out-file "./data/results/step2_res.txt" \
    --not-reliable $NOT_RELIABLE_COMMITS





