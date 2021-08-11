#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the AE to the running pipeline"
context="continuous-integration/98/esp-deploy"

post_state "$context" "$description" "$state"

if current_state
then
    # notify the pipeline that it is ready
    curl -XPOST "http://mentor0.jpl.nasa.gov:8080/app/submit?changeset=$(git rev-parse HEAD)&submission=now"
    sleep 10
fi

post_state "$context" "$description" "$state"
