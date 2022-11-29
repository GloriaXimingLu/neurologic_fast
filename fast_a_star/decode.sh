#!/usr/bin/env bash

export PYTHONPATH=/home/ximinglu/neurologic_fast

DATA_DIR='../dataset/commongen'
SPLIT='test'

DEVICES=$1
OUTPUT_FILE=generation/raw/${SPLIT}

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python decode_gpt2.py --model_name 'gpt2-large' \
  --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/reduce/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/reduce/constraint/${SPLIT}_key.constraint.json \
  --input_path ${DATA_DIR}/reduce/prompt/commongen.${SPLIT}.pt.txt \
  --concept_file ${DATA_DIR}/reduce/commongen.${SPLIT}.src_alpha.txt \
  --batch_size 16 --beam_size 20 --max_tgt_length 32 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 500000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha 0.175 --look_ahead_width 1 #--fusion_t 1.0