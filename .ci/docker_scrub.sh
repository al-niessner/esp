#! /usr/bin/env bash

docker rmi $(docker images | grep -v REPOSITORY | awk '{print $1 " " $2 " " $3}' | grep "<none>" | awk '{print $3}')
