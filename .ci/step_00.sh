#! /usr/bin/env bash

. .ci/util.sh

# https://developer.github.com/v3/repos/statuses/

state="pending" # "success" "pending" "failure" "error"
description="Build an environment for CI"
context="continuous-integration/00/environment"

post_state "$context" "$description" "$state"

if current_state
then
    lv="$(layer_versions)"
    baseVersion="${lv:0:16}"
    citVersion="${lv:17:16}"
    echo "Base layer $baseVersion"
    echo "CIT Version: $citVersion"

    if [ -z "$(docker images | awk -e '{print $1":"$2}' | grep esp_base:$baseVersion)" ]
    then
        echo "   Building base layer $baseVersion"
        docker build --network=host -t esp_base:${baseVersion} - < .ci/Dockerfile.base
    fi

    if [ -z "$(docker images | awk -e '{print $1":"$2}' | grep cit:$citVersion)" ]
    then
        echo "   Building CI Tools layer $citVersion"
        docker build --network=host -t esp_cit:${citVersion} - < .ci/Dockerfile.1
    fi

    rm .ci/Dockerfile.1
    state=`get_state`
fi

post_state "$context" "$description" "$state"
