#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=2 train.py --batch 4 --mention_inform --system_matching
# python3 -m torch.distributed.launch --nproc_per_node=2 train_final.py --batch 4 --mention_inform --system_matching