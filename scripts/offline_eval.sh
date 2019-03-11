#!/bin/sh
#####################
##### OUR OWN SETTING
#####################
export ROOT_DIR=~/mlp_project
export BASELINE_CODE_DIR=$ROOT_DIR/pointer-generator
export DATA_DIR=$ROOT_DIR/data/finished_files
export LOG_DIR=$ROOT_DIR/logs
# give a name to your experiment
export EXP_NAME=test

# VERY IMPORTANT
export POS_METHOD=no
export CHAR_METHOD=concate

##################
## EVAL CONCURRENT
##################
python $BASELINE_CODE_DIR/run_summarization.py\
    --mode=eval\
    --offline_eval=True\
    --data_path=$DATA_DIR/chunked/val_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD