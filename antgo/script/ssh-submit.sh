#!/usr/bin/env bash
username=$1
password=$2
ip=$3
gpu_num=$4
cpu_num=$5
memory_size=$6
command=$7
image=$8

# tar code 
tar -cf ../project.tar .
# # push to target machine
scp ../project.tar ${user}@${ip}:/home/${user}/
# # remote run
# ssh ${user}@${ip} < ssh-launch.sh ${user} ${cpu_num} ${memory_size} ${gpu_num} ${command} ${image}
