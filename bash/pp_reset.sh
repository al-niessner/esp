#! /usr/bin/env bash

DAWGIE_PIPELINE_HOST=${DAWGIE_PIPELINE_HOST:-${HOSTNAME:-excalibur.jpl.nasa.gov}}
DAWGIE_SSL_PEM_MYSELF=${DAWGIE_SSL_PEM_MYSELF:-${HOME}/.ssh/myself.pem}
let priv_port=${1:-${DAWGIE_FE_PORT:-8080}}+5

curl -XPOST --cert ${DAWGIE_SSL_PEM_MYSELF} "https://${DAWGIE_PIPELINE_HOST}:${priv_port}/app/reset&archive=${DAWGIE_ARCHIVE:-true}"
