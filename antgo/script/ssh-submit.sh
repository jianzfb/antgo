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

# 打包项目工程代码
# tar code 
tar -cf ../project.tar .

# 上传到所有目标机器
# push to target machine
target_ip_list=(${ip//,/ })
for target_ip in ${target_ip_list[@]}
do
   echo "push project code to "$target_ip
   ssh ${username}@${target_ip} "rm -rf /home/"${username}/${project}";mkdir /home/"${username}/${project}
   scp ../project.tar ${username}@${target_ip}:/home/${username}/${project}
   ssh ${username}@${target_ip} "cd /home/${username}/"${project}"/;tar -xf project.tar;rm project.tar;"
done

# 清理本地打包代码
# clear
rm ../project.tar

# 启动远程程序
# remote run
node_rank=0
for target_ip in ${target_ip_list[@]}
do
    script_folder=$( dirname "$0" ) 
    echo remote address ${username}@${target_ip}
    target_command=${command}" --node-rank="${node_rank}
    echo remote execute script ${target_command}
    echo remote image ${image}
    echo remote workspace /home/${username}/${project}
    echo remote running config cpu: ${cpu_num} memory: ${memory} gpu: ${gpu_num}
    ssh ${username}@${target_ip} 'bash -s' < ${script_folder}/ssh-launch.sh ${username} ${cpu_num} ${memory} ${gpu_num} \"${target_command}\" ${image} ${project}
    node_rank=$(expr $node_rank + 1)
done
