#! /usr/bin/env bash

myname="${DAWGIE_SSL_PEM_MYNAME:-excalibur.jpl.nasa.gov}"
myself="${DAWGIE_SSL_PEM_MYSELF:-${HOME}/.ssh/excalibur_identity.pem}"
let priv_port=${1:-${DAWGIE_FE_PORT:-8080}}+5

curl -XPOST --cert ${myself} "https://${myname}:${priv_port}/app/reset"  # &archive=${DAWGIE_ARCHIVE:-true}
