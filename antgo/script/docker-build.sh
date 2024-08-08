#!/usr/bin/env bash
cd ~/
tar -xf project.tar
cd {{project}}
docker build -t {{project}} .