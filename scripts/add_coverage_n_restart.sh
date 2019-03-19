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

############
## backup ##
############
echo "=========================="
echo "back up no coverage model.. hang on"
BACK_DIR="$LOG_DIR/$EXP_NAME"_backup
if [ ! -d "$BACK_DIR" ]; then
    cp -r "$LOG_DIR/$EXP_NAME" $BACK_DIR
fi
echo "=========================="


# start from the best model we've trained
echo "save best model ckpt to train"
frpg_run --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$VOCAB_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --restore_best_model=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD


# convert_to_coverage
# all args shall be the SAME as the model you load
frpg_run --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$VOCAB_DIR\
	--log_root=$LOG_DIR\
	--exp_name=$EXP_NAME\
    --coverage=True\
	--convert_to_coverage_model=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD

# check if we get the new ckpt
COVERAGE_CKPT=$(ls $LOG_DIR/$EXP_NAME/train| grep _cov_init)
if [ -z "$COVERAGE_CKPT" ]; then
    echo "cant found coverage ckpt..."
    exit
fi

########
## TRAIN
########
frpg_run --mode=train\
    --data_path=$DATA_DIR/chunked/train_*\
    --vocab_path=$VOCAB_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --coverage=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD & 

sleep 10

##################
## EVAL CONCURRENT
##################
frpg_run --mode=eval\
    --data_path=$DATA_DIR/chunked/val_*\
    --vocab_path=$VOCAB_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --coverage=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD & 