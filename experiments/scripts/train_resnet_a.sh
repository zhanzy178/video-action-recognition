#!/bin/bash
# Initial logs
# rm -rf ./experiments/logs/train_first_glance
mkdir ./experiments/logs
mkdir ./models
mkdir ./experiments/history

################## Train arguments ###############
# Train epoch
epoch=100
# Learning rate
lr=0.0001
# Weight decay
weight_decay=0.0001
# Batch size for train
batch_size=8
# momentum
momentum=0.9
# Number of classes
num=6
# Worker number
worker=7

################## Dataset arguments ###############
# Path to Images
ImagePath="data/PISC/image"
# Path to object boxes
ObjectsPath="data/objects/PISC_objects/"
# Path to test list
TrainList="data/list/PISC_fine_level_train.txt"
# Path to test list
TestList="data/list/PISC_fine_level_test.txt"


################## Record arguments ###############
# Path to save scores
ResultPath="experiments/logs/resnet_a"
# Print frequence
print_freq=100
# Dir to load/save model checkpoint
CheckpointDir="models/"
# File name
FileName="resnet_a"

CUDA_VISIBLE_DEVICES=0 python ./tools/train_resnet_a.py \
    $ImagePath \
    $ObjectsPath \
    $TrainList \
    $TestList \
    -n $num \
    -b $batch_size \
    --lr $lr \
    -m $momentum \
    --wd $weight_decay \
    -e $epoch \
    -j $worker \
    --print-freq $print_freq \
    --result-path $ResultPath \
    --checkpoint-dir $CheckpointDir \
    --checkpoint-name $FileName

