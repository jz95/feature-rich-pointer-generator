#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

#############
export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH
################

#####################
##### OUR OWN SETTING
#####################
export ROOT_DIR=/home/${STUDENT_ID}/mlp_project
export BASELINE_CODE_DIR=$ROOT_DIR/pointer-generator
# see the logs dir for experiment records
export DATA_DIR=${ROOT_DIR}/data
export LOG_DIR=${ROOT_DIR}/logs
# give a name to your experiment
export EXP_NAME=pos_concate

# VERY IMPORTANT
export POS_METHOD=concate
export CHAR_METHOD=no

###################
# LOAD BEST MODEL #
###################
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
echo "============================="
echo "save best model ckpt to train"
echo "============================="

python $BASELINE_CODE_DIR/run_summarization.py\
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
python $BASELINE_CODE_DIR/run_summarization.py\
    --mode=decode\
    --data_path=$DATA_DIR/chunked/test_*\
    --vocab_path=$DATA_DIR\
    --log_root=$LOG_DIR\
    --exp_name=$EXP_NAME\
    --single_pass=True\
    --how_to_use_pos=$POS_METHOD\
    --how_to_use_char=$CHAR_METHOD