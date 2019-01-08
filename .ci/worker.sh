#! /usr/bin/env bash

docker run \
       --rm \
       -e USER=$USER -e USERNAME=$USERNAME \
       --name ops_worker_$1 \
       --network host \
       -u $UID:$GROUPS \
       -v /proj/sdp/data:/proj/data -v ${HOME}/.gnupg:/proj/gnupg \
       esp_worker:latest
$(dirname $0)/crew.sh
