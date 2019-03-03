#!/bin/sh
#######################
## RESET YOUR ROOT PATH
#######################
export ROOT_DIR=~/mlp_project
export DATA_DIR=$ROOT_DIR/data/finished_files
export BASELINE_CODE_DIR=$ROOT_DIR/pointer-generator
# see the logs dir for experiment records
export LOG_DIR=$ROOT_DIR/logs
# give a name to your experiment
export EXP_NAME=pos_concate


###################
# LOAD BEST MODEL #
###################
echo "============================="
echo "save best model ckpt to train"
echo "============================="
python $BASELINE_CODE_DIR/run_summarization.py\
    --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --how_to_use_pos=concate\
    --restore_best_model=True

sleep 60

########
# DECODE
########
echo "============================="
echo "start decoding"
echo "============================="
python $BASELINE_CODE_DIR/run_summarization.py\
	--mode=decode\
	--data_path=$DATA_DIR/chunked/test_*\
	--vocab_path=$DATA_DIR\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_NAME\
    --how_to_use_pos=concate\
	--single_pass=True