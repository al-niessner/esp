#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Coverage analysis and checks of excalibur"
context="continuous-integration/04/esp-coverage"

post_state "$context" "$description" "$state"

if current_state
then
    state=`get_state`
fi

post_state "$context" "$description" "$state"
