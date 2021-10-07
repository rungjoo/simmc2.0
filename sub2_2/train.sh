#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=4 train_DDP.py --batch 4 --mention_inform --system_matching
# python3 -m torch.distributed.launch --nproc_per_node=4 train_DDP_final.py --batch 4 --mention_inform --system_matching