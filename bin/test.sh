#!/bin/bash

cd `dirname $0`/..

project=test
limit_batches=2
save_interval=2
batch_size=16
epoch=2
count=2
monotone28_checkpoint=ishiyama-k/pretrain/model-7wvag4lx:v1
color32_checkpoint=ishiyama-k/pretrain/model-phwzxk6r:v1

python="python3"
generate="$python src/wandb/generate"
generate_train="$generate/train.py"
generate_sweep="$generate/sweep.py"
pretrain="$python src/wandb/pretrain"
pretrain_train="$pretrain/train.py"
pretrain_sweep="$pretrain/sweep.py"
simclr="$python src/wandb/simclr"
simclr_train="$simclr/train.py"
simclr_sweep="$simclr/sweep.py"

training_base="--batch_size=$batch_size --epoch=$epoch --limit_batches=$limit_batches --save_interval=$save_interval --project=$project"
mnist="--dataset=mnist"
cifar10="--dataset=cifar10"
# monotone28="--backbone=monotone28 --backbone_checkpoint=$monotone28_checkpoint"
monotone28="--backbone=monotone28"
# color32="--backbone=color32 --backbone_checkpoint=$color32_checkpoint"
color32="--backbone=color32"
vit="--backbone=pretrained_vit_b_16"
resnet="--backbone=resnet18"
dino="--backbone=dino_s_16"
gumbel="--gumbel_softmax --wo_policy_loss --entropy_coeff=0"
sweep_base="--project=$project --count=$count --limit_batches=$limit_batches --epoch=$epoch"

commands=(
"$generate_train $training_base $mnist"
"$generate_train $training_base $cifar10"
"$generate_train $training_base $mnist $monotone28"
"$generate_train $training_base $cifar10 $color32"
"$generate_train $training_base $cifar10 $vit"
"$generate_train $training_base $cifar10 $resnet"
"$generate_train $training_base $cifar10 $dino"
"$generate_train $training_base $mnist --data_augmentation"
"$generate_train $training_base $mnist --classifier"
"$generate_train $training_base $mnist $gumbel"
"$generate_sweep $sweep_base"
"$pretrain_train $training_base $mnist"
"$pretrain_train $training_base $cifar10"
"$pretrain_train $training_base $cifar10 $vit"
"$pretrain_train $training_base --dataset=mnist --data_augmentation"
"$pretrain_sweep $sweep_base"
"$simclr_train $training_base $mnist"
"$simclr_train $training_base $cifar10"
"$simclr_train $training_base $mnist $monotone28"
"$simclr_train $training_base $cifar10 $color32"
"$simclr_train $training_base $cifar10 $vit"
"$simclr_train $training_base $cifar10 $resnet"
"$simclr_train $training_base $cifar10 $dino"
"$simclr_train $training_base $mnist --data_augmentation"
"$simclr_train $training_base $mnist --classifier"
"$simclr_train $training_base $mnist $gumbel"
"$simclr_train $training_base $mnist --lazy_speaker"
"$simclr_sweep $sweep_base"
)

mkdir -p outputs/test_results
parallel -j 8 --bar \
    --results outputs/test_results/{#} \
    --joblog outputs/test_results/joblog.txt \
    ::: "${commands[@]}"

awk '$7!=0 {print "Job", $1, "with exit code", $7, "failed. See output in outputs/test_results/" $1 ".err"}' outputs/test_results/joblog.txt