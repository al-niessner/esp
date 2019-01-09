#! /usr/bin/env bash

while [ -f /tmp/crew.ops ]
do
    for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
    do
        existance="$(docker ps -a | grep ops_worker_$i)"

        if [ -z "${existance}" ]
        then
            $(dirname $0)/worker.sh $i
            docker ps | grep ops_worker_$i
        fi
    done
    sleep 5
done
