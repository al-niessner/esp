#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the new AE (worker)"
context="continuous-integration/01/esp-deploy"

post_state "$context" "$description" "$state"

if current_state
then
    curl -XPOST http://mentor.jpl.nasa.gov:8080/app/submit?changeset=$(git rev-parse HEAD)\&submission=now > /dev/null 2>&1
    state=`get_state`
fi

post_state "$context" "$description" "$state"
