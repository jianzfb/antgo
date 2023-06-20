#!/usr/bin/env bash
if [ ! -d "~/.ssh" ];then
    mkdir ~/.ssh
fi
cd ~/.ssh
rsa='placeholder'
echo ${rsa} >> authorized_keys