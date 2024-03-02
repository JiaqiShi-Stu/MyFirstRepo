#!/bin/bash

export PYTHONPATH=${PYTHONPATH}:$(pwd)


CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=7310 DiffIR/train.py -opt options/train_DiffIRS1_x4_SD300.yml --launcher pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=7310 DiffIR/train.py -opt options/train_DiffIRS1_x2.yml --launcher pytorch 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=7310 DiffIR/train.py -opt options/train_DiffIRS1_x1.yml --launcher pytorch 


#  python scripts/generate_meta_info.py --input /data/sjq/SR/SRdataset/EXP_data/SD300/train/HR \
#  --root /data/sjq/SR/SRdataset/EXP_data/SD300/train/HR \
#  --meta_info /data/sjq/SR/SRdataset/EXP_data/SD300_ORI/SAVE/train.txt