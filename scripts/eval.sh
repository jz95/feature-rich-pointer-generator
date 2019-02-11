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
export EXP_NAME=myexperiment

########################
## RUN EVAL CONCURRENTLY
## USING THE SAME SETTING
## AS TRAINING
########################
python $BASELINE_CODE_DIR/run_summarization.py\
	--mode=eval\
	--data_path=$DATA_DIR/chunked/val_*\
	--vocab_path=$DATA_DIR/vocab\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_NAME