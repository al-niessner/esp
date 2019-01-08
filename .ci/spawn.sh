#! /usr/bin/env bash

if [ -f /tmp/crew.ops ]
then
    for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16
    do
        if [ -z "$(docker ps | grep ops_worker_$i)" ]
        then
            $(dirname $0)/worker.sh $i &
        fi
    done
fi
