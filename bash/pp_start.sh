#! /usr/bin/env bash

DAWGIE_SSL_PEM_MYNAME=excalibur.jpl.nasa.gov
DAWGIE_SSL_PEM_MYSELF=/proj/myself/excalibur_identity.pem

latest_tag=$(docker images | grep esp_devel | head -n 1 | awk '{print $2}')
echo "Starting excalibur:"
echo "   tag:  ${TAG:-${latest_tag}}"
echo "   port: ${1:-${DAWGIE_FE_PORT:-9990}}"
docker run --detach \
       -e EXCALIBUR_LEVER_AND_KNOB_SETTINGS=/proj/data/runtime/${USER}.xml \
       -e USER=${USER} \
       -e USERNAME=${USERNAME} \
       -e DAWGIE_DB_NAME=$USER \
       -e DAWGIE_SSL_PEM_FILE=/etc/ssl/server.pem \
       --name ${USER}_privatepl \
       --network host \
       --rm \
       -u ${UID}:1512 \
       -v ${HOME}/.ssh:/proj/myself:ro \
       -v /etc/ssl:/etc/ssl:ro \
       -v /proj/sdp/data:/proj/data \
       -v ${HOME}/.theano:/proj/data/.theano \
       -v /proj/sdp/data/logs:/proj/logs \
       -v /proj/sdp/data/${USER}/db:/proj/data/db \
       -v ${HOME}/.gnupg:/proj/data/certs \
       -v $(realpath $(dirname $0)/..):/proj/src/ae \
       esp_devel:${TAG:-${latest_tag}} \
       python3 -m dawgie.pl -p ${1:-${DAWGIE_FE_PORT:-9990}} -l ${USER}.log -L 20
