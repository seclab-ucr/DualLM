#!/bin/bash

CURRENT_DIR=$(pwd)
HOME_DIR=$(realpath ../)
DATA_HOME=${HOME_DIR}/data
MODEL_HOME="../models"
#
 
MODEL_DIR=${MODEL_HOME}/
DATA_DIR=${DATA_HOME}/binarize_step2
NUM_CLASSES=3
res_file=step2_res
 
 
FINTUNE_DATA=${DATA_HOME}/ #xxx should be the same with name used in build_eval_data_for_random_given()
 
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=$1
# mkdir -p ${HOME_DIR}/results/${FUNETUNE_MODEL}


python  bert_res.py \
--model_dir $MODEL_DIR \
--model_name step2.pt \
--data_bin_path $DATA_DIR \
--input_file ${FINTUNE_DATA}/spm_preprocess/eval.spm \
--label_file ${FINTUNE_DATA}/train_valid_test/eval.label \
--batch_size 128 \
--max_example -1 \
--num_classes $NUM_CLASSES \
--output ${HOME_DIR}/data/results/${res_file}.txt > /dev/null 2>&1

