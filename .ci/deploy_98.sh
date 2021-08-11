#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the new AE (base, cit, devel, server, tools, worker)"
context="continuous-integration/99/esp-deploy"

post_state "$context" "$description" "$state"

if current_state
then
    # notify the pipeline that it is ready
    curl -XPOST http://mentor.jpl.nasa.gov:8080/app/submit?changeset=$(git rev-parse HEAD)\&submission=now > /dev/null 2>&1
    sleep 10
fi

post_state "$context" "$description" "$state"
