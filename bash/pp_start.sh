#! /usr/bin/env bash

latest_tag=$(docker images | grep esp_devel | head -n 1 | awk '{print $2}')
myname="${DAWGIE_SSL_PEM_MYNAME:-excalibur.jpl.nasa.gov}"
myself="${DAWGIE_SSL_PEM_MYSELF:-/proj/data/certs/excalibur_identity.pem}"
echo "Starting excalibur:"
echo "   tag:  ${TAG:-${latest_tag}}"
echo "   port: ${1:-${DAWGIE_FE_PORT:-9990}}"
docker run --detach \
       -e DAWGIE_DB_NAME=$USER \
       -e DAWGIE_SSL_PEM_FILE=/etc/ssl/server.pem \
       -e DAWGIE_SSL_PEM_MYNAME="$myname"\
       -e DAWGIE_SSL_PEM_MYSELF="$myself" \
       -e EXCALIBUR_LEVER_AND_KNOB_SETTINGS=/proj/data/runtime/${USER}.xml \
       -e USER=${USER} \
       -e USERNAME=${USERNAME} \
       --name ${USER}_privatepl \
       --network host \
       --rm \
       -u ${UID}:1512 \
       -v /etc/ssl:/etc/ssl:ro \
       -v /usr/local/share/ca-certificates:/usr/local/share/ca-certificates:ro \
       -v /usr/share/ca-certificates:/usr/share/ca-certificates:ro \
       -v /proj/sdp/data:/proj/data \
       -v ${HOME}/.theano:/proj/data/.theano \
       -v /proj/sdp/data/logs:/proj/logs \
       -v /proj/sdp/data/${USER}/db:/proj/data/db \
       -v ${HOME}/.ssh:/proj/data/certs \
       -v $(realpath $(dirname $0)/..):/proj/src/ae \
       esp_devel:${TAG:-${latest_tag}} \
       python3 -m dawgie.pl -p ${1:-${DAWGIE_FE_PORT:-9990}} -l ${USER}.log -L 20
