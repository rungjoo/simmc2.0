#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=4 train.py --batch 4 --background --post_back --post --post_balance 'all' --utt_category --system_matching --system_train
# python3 -m torch.distributed.launch --nproc_per_node=4 train_final.py --batch 4 --background --post_back --post --post_balance 'all' --utt_category --system_matching --system_train