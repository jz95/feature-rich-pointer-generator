#!/bin/sh
#######################
## RESET YOUR ROOT PATH
#######################
export ROOT_DIR=~/mlp_project
export DATA_DIR=$ROOT_DIR/data/finished_files
export BASELINE_CODE_DIR=$ROOT_DIR/pointer-generator
export LOG_DIR=$ROOT_DIR/logs
export EXP_DIR=$ROOT_DIR/exps
export EXP_NAME=myexperiment

########
## TRAIN
########
python $BASELINE_CODE_DIR/run_summarization.py\
	--mode=train\
	--data_path=$DATA_DIR/chunked/train_*\
	--vocab_path=$DATA_DIR/vocab\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_DIR/$EXP_NAME

########
## EVAL
########
# python $BASELINE_CODE_DIR/run_summarization.py\
# 	--mode=eval\
# 	--data_path=$DATA_DIR/chunked/val_*\
# 	--vocab_path=$DATA_DIR/vocab\
# 	--log_root=$LOG_DIR\
# 	--exp_name=$EXP_DIR/$EXP_NAME

#########
## DECODE
#########
# python $BASELINE_CODE_DIR/run_summarization.py\
# 	--mode=decode\
# 	--data_path=$DATA_DIR/chunked/val_*\
# 	--vocab_path=$DATA_DIR/vocab\
# 	--log_root=$LOG_DIR\
# 	--exp_name=$EXP_DIR/$EXP_NAME