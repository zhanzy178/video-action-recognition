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
lr=0.01
# Weight decay
weight_decay=0.0001
# Batch size for train
batch_size=4
# momentum
momentum=0.9
# Number of classes
num=400
# Worker number
worker=7

################## Dataset arguments ###############
Dataset="KineticsDataset_a"
# Directory to HMDB51 video
VideoDir="data/kinetics/Kinetics_trimmed_videos"
# Directory to HMDB51 frame
FrameDir="data/kinetics/frame"
# Path to HMDB51 51 class meta information.
MetaPath="data/kinetics/meta.txt"
# Path to HMDB51 train list
TrainListPath="data/kinetics/kinetics_train/kinetics_train.csv"
# Path to HMDB51 test list
TestListPath="data/kinetics/kinetics_val/kinetics_val.csv" 

# Number of frames that extract from video.
num_frame=25
# Refresh flag for clearing frames and create new one.
refresh=0


################## Record arguments ###############
# Path to save scores
ResultPath="experiments/logs/kinetics_resnet_a"
# Print frequence
print_freq=100
# Dir to load/save model checkpoint
CheckpointDir="models/"
# File name
FileName="kinetics_resnet_a"

CUDA_VISIBLE_DEVICES=6,7 python ./tools/train_resent_a.py \
    $Dataset \
    $VideoDir \
    $FrameDir \
    $MetaPath \
    $TrainListPath \
    $TestListPath \
    --num-frame $num_frame \
    --refresh $refresh \
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

