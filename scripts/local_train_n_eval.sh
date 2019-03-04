#!/bin/sh
#####################
##### OUR OWN SETTING
#####################
export ROOT_DIR=~/mlp_project
export BASELINE_CODE_DIR=$ROOT_DIR/pointer-generator
export DATA_DIR=$ROOT_DIR/data/finished
export LOG_DIR=$ROOT_DIR/logs
# give a name to your experiment
export EXP_NAME=pos_concate

# VERY IMPORTANT
export POS_METHOD=concate
export CHAR_METHOD=no

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python $BASELINE_CODE_DIR/run_summarization.py\
    --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD & 

sleep 600   # no hurry to run eval 

##################
## EVAL CONCURRENT
##################
python $BASELINE_CODE_DIR/run_summarization.py\
    --mode=eval\
    --data_path=$DATA_DIR/chunked/val_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD & 