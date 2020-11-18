#!/usr/bin/env bash
export PATH=$PATH:/usr/local/cuda-8.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-8.0/lib64
export GIL_HOME=$(pwd)
export LOG_DIR=$GIL_HOME/logs
export PYTHONPATH=$GIL_HOME:$PYTHONPATH
export DATAPATH=$GIL_HOME/data
source activate hy-torch