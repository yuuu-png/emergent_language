#!/bin/bash

cd `dirname $0`/..

dims=(32 64 128 256 512)

for dim in "${dims[@]}"; do
    python3 src/wandb/pretrain/train.py --dataset=cifar10 --backbone=color32 --hidden_dim=$dim --save_interval=100 --images_per_batch=100 --data_augmentation
done
