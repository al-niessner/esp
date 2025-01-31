#! /usr/bin/env bash

docker run \
       --detach \
       --init \
       --rm \
       -e DAWGIE_SSL_PEM_MYNAME=excalibur.jpl.nasa.gov \
       -e DAWGIE_SSL_PEM_MYSELF=/proj/mycerts/ops.pem \
       -e USER=$USER -e USERNAME=$USERNAME \
       --name ops_worker_$1 \
       --network host \
       -u $UID:$GROUPS \
       -v /proj/sdp/data:/proj/data \
       -v ${HOME}/.ssh:/proj/mycerts:ro \
       -v /etc/ssl:/etc/ssl:ro \
       -v /usr/local/share/ca-certificates:/usr/local/share/ca-certificates:ro \
       -v /usr/share/ca-certificates:/usr/share/ca-certificates:ro 
       esp_worker:latest -i $1 -n excalibur.jpl.nasa.gov -p 8081 > /dev/null
