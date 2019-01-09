#! /usr/bin/env bash

while [ -f /tmp/crew.ops ]
do
    for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
    do
        if [ -z "$(docker ps | grep ops_worker_$i)" ]
        then
            $(dirname $0)/worker.sh $i
            docker ps | grep ops_worker_$i
            while [ -z "$([ -f /tmp/crew.ops ]  && ( docker ps | grep ops_worker_$i ) || echo abort)" ]
            do
                echo "waiting for ops_worker_$i to dawn"
                sleep 2
            done
        fi
    done
    echo 'sleeping until next check'
    sleep 5
done
