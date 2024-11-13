#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Unit testing of excalibur"
context="continuous-integration/03/esp-pytest"

post_state "$context" "$description" "$state"

if current_state
then
    docker run -e USER=$USER --rm -v $PWD:$PWD -u $UID -w $PWD esp_cit:$(cit_version) python3 -m pytest --cov=excalibur $PWD/test | tee unittest.rpt.txt
    [ 0 -lt `grep FAILED unittest.rpt.txt | wc -l` ]  && echo 'failure' > .ci/status.txt
    state=`get_state`
fi

post_state "$context" "$description" "$state"
current_state
