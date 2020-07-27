#!/usr/bin/env bash
if [ $# -eq 0 ]
then
    printf "Please Set Run Command train/challenge\n"
    printf "For Example: run.sh train\n"
elif [ $# -eq 1 ]
then
    if [ $1 = "help" ]
    then
        printf "parameter 1: train/challenge\n"
        printf "parameter 2: experiemnt_id (allow model to recover from experiment record)\n"
    elif [[ $1 = "train" || $1 = "challenge" ]]
    then
        
        nohup antgo $1 --main_file=FCN_main.py --main_param=FCN_param.yaml --task=FCN_task.xml> run.log 2>&1 &
        
    else
        printf "only support train/challenge command\n"
    fi
elif [ $# -eq 2 ]
then
    if [[ $1 = "train" || $1 = "challenge" ]]
    then
        
        nohup antgo $1 --main_file=FCN_main.py --main_param=FCN_param.yaml --task=FCN_task.xml --from_experiment=$2> run.log 2>&1 &
        
    else
        printf "only support train/challenge command\n"
    fi
fi