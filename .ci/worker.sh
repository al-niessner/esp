#! /usr/bin/env bash

docker run \
       --detach \
       --rm \
       -e USER=$USER -e USERNAME=$USERNAME \
       --name ops_worker_$1 \
       --network host \
       -u $UID:$GROUPS \
       -v /proj/sdp/data:/proj/data -v ${HOME}/.gnupg:/proj/gnupg \
       esp_worker:latest > /dev/null
