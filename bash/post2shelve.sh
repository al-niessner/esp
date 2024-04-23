#! /usr/bin/env bash

latest_tag=$(docker images | grep esp_devel | head -n 1 | awk '{print $2}')
docker run \
       -e USER=$USER \
       -e USERNAME=$USERNAME \
       --network host \
       --rm \
       -u $UID \
       -v /proj/sdp/data:/proj/data \
       esp_tools:${TAG:-$latest_tag} \
       dawgie.db.tools.post2shelve -O /proj/data/${USER}/db -p $USER
