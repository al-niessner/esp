#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Coding standard checks of the excalibur software"
context="continuous-integration/01/esp-PEP8"

post_state "$context" "$description" "$state"

if current_state
then
    docker run --rm -v $PWD:$PWD -u $UID -w $PWD esp_cit:$(cit_version) pycodestyle \
           --ignore=E24,E121,E123,E124,E126,E127,E211,E225,E226,E231,E252,E301,E302,E305,E402,E501,W504,E701,E702,E704,E722,E741 \
           --exclude=.ci/Dockerfile.py \
           --statistics ${PWD} | tee ${PWD}/pep8.rpt.txt
    errs=`wc -l pep8.rpt.txt | awk '{print $1}'`
    [ $errs -ne 0 ] && echo -n "failure" > .ci/status.txt
    state=`get_state`
fi

post_state "$context" "$description" "$state"
git checkout .ci/status.txt
