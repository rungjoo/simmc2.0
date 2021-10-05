#!/bin/bash
python -m gpt2_dst.scripts.evaluate \
  --input_path_target="./gpt2_dst/data/simmc2_dials_dstc10_devtest_target.txt" \
  --input_path_predicted="../../../sub2/results/dstc10-simmc-entry/dstc10-simmc-teststd-pred-subtask-3.txt" \
  --output_path_report='result.txt'