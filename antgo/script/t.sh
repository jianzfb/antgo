#!/bin/bash
 
# string="hello,shell,split,test"  
# array=(${string//,/ })  
# for var in ${array[@]}
# do
#    echo $var
# done 

ip="a,b,c"
target_ip_list=(${ip//,/ })
node_rank=0
for target_ip in ${target_ip_list[@]}
do
   echo "push project code to "$target_ip
   node_rank=$(expr $node_rank + 1)
   echo "node-rank "${node_rank}
done
