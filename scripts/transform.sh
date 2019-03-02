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
export EXP_NAME=test_run

########
## TRAIN
########
python $BASELINE_CODE_DIR/run_summarization.py\
	--mode=train\
	--data_path=$DATA_DIR/chunked/train_*\
	--vocab_path=$DATA_DIR\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_NAME\
	--how_to_use_pos=no\
    --coverage=True\
	--convert_to_coverage_model=True

