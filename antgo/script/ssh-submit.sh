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
env=${10}
submit_time=${11}
data_folder=${12}
project_folder=${13}

if [ -z "${project_folder}" ]; then
    project_folder=/home/${username}
fi


# 打包项目工程代码
# tar code 
tar -cf ../project.tar .

# 上传到所有目标机器（并行）
# push to target machine
target_ip_list=(${ip//,/ })
uploadProject(){
    # username, target_ip, project, submit_time, project_folder
    # 1,        2,         3,       4,           5
    echo "push project code to "${4}/${3}" (in "$2 ")"
    ssh ${1}@${2} "mkdir -p "${5}/${4}/${3}
    scp ../project.tar ${1}@${2}:${5}/${4}/${3}
    ssh ${1}@${2} "cd ${5}/${4}/"${3}"/;tar -xf project.tar;rm project.tar;"
}

pidarr=()
for target_ip in ${target_ip_list[@]}
do
    echo ${target_ip}
    uploadProject ${username} ${target_ip} ${project} ${submit_time} ${project_folder} &
    pid=$!
    pidarr+=(${pid})
done

for pid in "${pidarr[@]}"; do
    wait $pid
done


# 清理本地打包代码
# clear
rm ../project.tar

executeProject(){
    # username, target_ip, project, command, image, env, cpu_num, gpu_num, memory, script_folder, submit_time, data_folder, project_folder
    # 1,        2,         3,       4,       5,     6,    7,      8,       9,      10,            11,          12,          13
    echo remote address ${1}@${2}
    echo remote execute script ${4}
    echo remote image ${5}
    echo remote workspace ${13}/${3}
    echo remote running config cpu: ${7} memory: ${9} gpu: ${8}
    echo remote data folder: ${12}
    echo remote project folder: ${13}
    echo execute ${10}/ssh-launch.sh ${1} ${7} ${9} ${8} \"${4}\" ${5} ${3} ${6}
    ssh ${1}@${2} 'bash -s' < ${10}/ssh-launch.sh ${1} ${7} ${9} ${8} \"${4}\" ${5} ${3} ${6} ${11} ${12} ${13}
}

# 启动远程程序
# remote run
node_rank=0
pidarr=()
for target_ip in ${target_ip_list[@]}
do
    script_folder=$( dirname "$0" )
    remote_comand=${command}" --node-rank="${node_rank}
    executeProject ${username} ${target_ip} ${project} "${remote_comand}" ${image} ${env} ${cpu_num} ${gpu_num} ${memory} ${script_folder} ${submit_time} ${data_folder} ${project_folder} &
    node_rank=$(expr $node_rank + 1)

    pid=$!
    pidarr+=(${pid})
done

for pid in "${pidarr[@]}"; do
    wait $pid
done
