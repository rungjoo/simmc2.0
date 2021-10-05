#!/bin/bash

# Train split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../../data/simmc2_dials_dstc10_train.json" \
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_train_predict.txt" \
    --output_path_target="./gpt2_dst/data/simmc2_dials_dstc10_train_target.txt" \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json"
    
# Dev split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../../data/simmc2_dials_dstc10_dev.json" \
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_dev_predict.txt" \
    --output_path_target="./gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt" \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --input_path_retrieval="../../data/simmc2_dials_dstc10_dev_retrieval_candidates.json" \
    --output_path_retrieval="./gpt2_dst/data/dev_retrieval.txt" \

# # Devtest split
python3 -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../../data/simmc2_dials_dstc10_devtest.json" \
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_devtest_predict.txt" \
    --output_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --input_path_retrieval="../../data/simmc2_dials_dstc10_devtest_retrieval_candidates.json" \
    --output_path_retrieval="./gpt2_dst/data/devtest_retrieval.txt" \