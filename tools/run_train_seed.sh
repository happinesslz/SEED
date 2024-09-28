#! /bin/bash

### training 100% data for 6 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_small.yaml --extra_tag seed_with_second_waymo_small_full_data_ep6 \
--batch_size 24 --epochs 6 --max_ckpt_save_num 4 --sync_bn


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_base.yaml --extra_tag seed_with_second_waymo_base_full_data_ep6 \
--batch_size 24 --epochs 6 --max_ckpt_save_num 4 --sync_bn


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_large.yaml --extra_tag seed_with_second_waymo_small_full_data_ep6 \
--batch_size 24 --epochs 6 --max_ckpt_save_num 4 --sync_bn


### training 100% data for 12 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_small.yaml --extra_tag seed_with_second_waymo_small_full_data_ep12 \
--batch_size 24 --epochs 12 --max_ckpt_save_num 4 --sync_bn


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_base.yaml --extra_tag seed_with_second_waymo_base_full_data_ep12 \
--batch_size 24 --epochs 12 --max_ckpt_save_num 4 --sync_bn


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  NCCL_IB_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=8 train.py --tcp_port 29566  --launcher pytorch  \
--cfg_file ./cfgs/seed_models/seed_with_second_waymo_large.yaml --extra_tag seed_with_second_waymo_small_full_data_ep12 \
--batch_size 24 --epochs 12 --max_ckpt_save_num 4 --sync_bn


