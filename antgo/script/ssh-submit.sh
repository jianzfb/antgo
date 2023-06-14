#!/usr/bin/env bash
username=$1
password=$2
ip=$3
gpu_num=$4
cpu_num=$5
memory=$6 
command=$7
image=$8
project=$9

# tar code 
tar -cf ../project.tar .
# push to target machine
ssh ${username}@${ip} "rm -rf /home/"${username}/${project}";mkdir /home/"${username}/${project}
scp ../project.tar ${username}@${ip}:/home/${username}/${project}
ssh ${username}@${ip} "cd /home/${username}/"${project}"/;tar -xf project.tar;rm project.tar;"
# clear
rm ../project.tar

# remote run
script_folder=$( dirname "$0" ) 
echo remote address ${username}@${ip}
echo remote execute script ${command}
echo remote image ${image}
echo remote workspace /home/${username}/${project}
echo remote running config cpu: ${cpu_num} memory: ${memory} gpu: ${gpu_num}
ssh ${username}@${ip} 'bash -s' < ${script_folder}/ssh-launch.sh ${username} ${cpu_num} ${memory} ${gpu_num} \"${command}\" ${image} ${project}

