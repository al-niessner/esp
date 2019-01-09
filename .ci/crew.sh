#! /usr/bin/env bash

while [ -f /tmp/crew.ops ]
do
    for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
    do
        existance="$(docker ps -a | grep ops_worker_$i)"
        echo "existance of ops_worker_${i}: ${existance}"

        if [ -z "${existance}" ]
        then
            $(dirname $0)/worker.sh $i
            docker ps | grep ops_worker_$i
        fi
    done
    echo 'sleeping until next check'
    sleep 5
done
