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
num=51
# Worker number
worker=7

################## Dataset arguments ###############
# Directory to HMDB51 video
VideoDir="data/HMDB51/video"
# Directory to HMDB51 frame
FrameDir="data/HMDB51/frame"
# Path to HMDB51 51 class meta information.
MetaPath="data/HMDB51/meta.txt"
# Path to HMDB51 train list
TrainListPath="data/HMDB51/train_list.txt" 
# Path to HMDB51 test list
TestListPath="data/HMDB51/test_list.txt" 

# Two stream object list path
TrainObjectListPath='data/HMDB51/train_bbox_list.pth'
TestObjectListPath='data/HMDB51/test_bbox_list.pth'

# Number of frames that extract from video.
num_frame=3
num_object=5
# Refresh flag for clearing frames and create new one.
refresh=0


################## Record arguments ###############
# Path to save scores
ResultPath="experiments/logs/resnet_dual"
# Print frequence
print_freq=100
# Dir to load/save model checkpoint
CheckpointDir="models/"
# File name
FileName="resnet_dual"

CUDA_VISIBLE_DEVICES=6,7 python ./tools/train_resnet_dual.py \
    $VideoDir \
    $FrameDir \
    $MetaPath \
    $TrainListPath \
    $TestListPath \
    $TrainObjectListPath \
    $TestObjectListPath \
    --num-frame $num_frame \
    --num-object $num_object \
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

