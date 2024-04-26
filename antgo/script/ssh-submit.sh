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


# 打包项目工程代码
# tar code 
tar -cf ../project.tar .

# 上传到所有目标机器（并行）
# push to target machine
target_ip_list=(${ip//,/ })
uploadProject(){
    # username, target_ip, project
    # 1,        2,         3
    echo "push project code to "$3"/"$2
    ssh ${1}@${2} "rm -rf /home/"${1}/${3}";mkdir /home/"${1}/${3}
    scp ../project.tar ${1}@${2}:/home/${1}/${3}
    ssh ${1}@${2} "cd /home/${1}/"${3}"/;tar -xf project.tar;rm project.tar;"
}

pidarr=()
for target_ip in ${target_ip_list[@]}
do
    uploadProject ${username} ${target_ip} ${project} &
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
    # username, target_ip, project, command, image, env, cpu_num, gpu_num, memory, script_folder
    # 1,        2,         3,       4,       5,     6,    7,      8,       9,      10
    echo remote address ${1}@${2}
    echo remote execute script ${4}
    echo remote image ${5}
    echo remote workspace /home/${1}/${3}
    echo remote running config cpu: ${7} memory: ${9} gpu: ${8}
    echo execute ${10}/ssh-launch.sh ${1} ${7} ${9} ${8} \"${4}\" ${5} ${3} ${6}
    ssh ${1}@${2} 'bash -s' < ${10}/ssh-launch.sh ${1} ${7} ${9} ${8} \"${4}\" ${5} ${3} ${6}
}

# 启动远程程序
# remote run
node_rank=0
pidarr=()
for target_ip in ${target_ip_list[@]}
do
    script_folder=$( dirname "$0" )
    remote_comand=${command}" --node-rank="${node_rank}
    executeProject ${username} ${target_ip} ${project} "${remote_comand}" ${image} ${env} ${cpu_num} ${gpu_num} ${memory} ${script_folder} &
    node_rank=$(expr $node_rank + 1)

    pid=$!
    pidarr+=(${pid})
done

for pid in "${pidarr[@]}"; do
    wait $pid
done
