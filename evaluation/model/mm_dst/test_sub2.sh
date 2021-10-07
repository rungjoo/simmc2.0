#!/bin/bash
python -m gpt2_dst.scripts.evaluate \
  --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
  --input_path_predicted="../../../sub2/results/dstc10-simmc-entry/dstc10-simmc-teststd-pred-subtask-3_1.txt" \
  --output_path_report='result1.txt'
  
  python -m gpt2_dst.scripts.evaluate \
  --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
  --input_path_predicted="../../../sub2/results/dstc10-simmc-entry/dstc10-simmc-teststd-pred-subtask-3_2.txt" \
  --output_path_report='result2.txt'
  
# python -m gpt2_dst.scripts.evaluate \
#   --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
#   --input_path_predicted="../../../../sub2_split_mod2_2/results/post_use_all_mention_inform_use_utt_category_use_system_matching_use_system_train_use_background_use_post_back_use_same_dstc.txt" \
#   --output_path_report='result.txt'  
  
# python -m gpt2_dst.scripts.evaluate \
#   --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
#   --input_path_predicted="/data/project/rw/rung/00_company/03_DSTC10_SIMMC/sub2_split_mod2_2/roberta-large_models/post_use_all/sigmoid/context/unbalance/mention_inform_no_use/utt_category_use/meta_no_use/system_matching_use/system_train_use/background_use/post_back_use/dstc_test1.txt" \
#   --output_path_report='result2.txt'    
  
  