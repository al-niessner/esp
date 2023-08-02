#! /usr/bin/env bash

docker run \
       --detach \
       --init \
       --rm \
       -e USER=$USER -e USERNAME=$USERNAME \
       --name ops_worker_$1 \
       --network host \
       -u $UID:$GROUPS \
       -v /proj/sdp/data:/proj/data -v ${HOME}/.gnupg:/proj/gnupg \
       esp_worker:latest -i $1 -n mentor0.jpl.nasa.gov -p 8081 > /dev/null
