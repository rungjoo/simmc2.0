#!/bin/bash
python -m gpt2_dst.scripts.evaluate \
  --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
  --input_path_predicted="../../../sub2_12/results/dstc10-simmc-entry/dstc10-simmc-devtest-pred-subtask-3_1.txt" \
  --output_path_report='result1.txt'
  
python -m gpt2_dst.scripts.evaluate \
--input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
--input_path_predicted="../../../sub2_12/results/dstc10-simmc-entry/dstc10-simmc-devtest-pred-subtask-3_2.txt" \
--output_path_report='result2.txt'
  
python -m gpt2_dst.scripts.evaluate \
--input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
--input_path_predicted="../../../sub2_3/results/dstc10-simmc-entry/dstc10-simmc-devtest-pred-subtask-3_3.txt" \
--output_path_report='result3.txt'
  
python -m gpt2_dst.scripts.evaluate \
--input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
--input_path_predicted="../../../sub2_4/results/dstc10-simmc-entry/dstc10-simmc-devtest-pred-subtask-3_4.txt" \
--output_path_report='result4.txt'   
  