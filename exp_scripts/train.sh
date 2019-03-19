#!/bin/sh
#####################
##### OUR OWN SETTING
#####################
export ROOT_DIR=~/mlp_project
export DATA_DIR=$ROOT_DIR/data/finished_files
export VOCAB_DIR=$ROOT_DIR/vocab
export LOG_DIR=$ROOT_DIR/logs
# give a name to your experiment
export EXP_NAME=test

# VERY IMPORTANT
export POS_METHOD=concate
export CHAR_METHOD=concate

# Activate the relevant virtual environment:

frpg_run --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$VOCAB_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD

# sleep 60   # no hurry to run eval 

# ##################
# ## EVAL CONCURRENT
# ##################
# frpg_run --mode=eval\
#     --data_path=$DATA_DIR/chunked/val_*\
#     --vocab_path=$DATA_DIR\
#     --log_root=$LOG_DIR\
#     --exp_name=$EXP_NAME\
#     --how_to_use_pos=$POS_METHOD\
#     --how_to_use_char=$CHAR_METHOD & 
