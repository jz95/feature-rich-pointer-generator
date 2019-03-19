#!/bin/sh
#######################
## RESET YOUR ROOT PATH
#######################
export ROOT_DIR=~/mlp_project
export DATA_DIR=$ROOT_DIR/data/finished_files
export VOCAB_DIR=$ROOT_DIR/vocab
export LOG_DIR=$ROOT_DIR/logs
# give a name to your experiment
export EXP_NAME=test

# VERY IMPORTANT
export POS_METHOD=concate
export CHAR_METHOD=concate

###################
# LOAD BEST MODEL #
###################
echo "============================="
echo "save best model ckpt to train"
echo "============================="
frpg_run
    --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --restore_best_model=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD


sleep 60

########
# DECODE
########
echo "============================="
echo "start decoding"
echo "============================="
frpg_run
	--mode=decode\
	--data_path=$DATA_DIR/chunked/test_*\
	--vocab_path=$DATA_DIR\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_NAME\
	--single_pass=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD