#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Build a server to manage the AE"
context="continuous-integration/01/esp-build-server"

post_state "$context" "$description" "$state"

if current_state
then
    lv=$(layer_versions)
    baseVersion="${lv:0:16}"
    echo "base version: ${baseVersion}"

    if [ -z "$(docker images | grep esp_server:${baseVersion})" ]
    then
        python3 <<EOF
with open ('.ci/Dockerfile.server', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.1', 'tw') as f: f.write (text.replace ("ghrVersion", "${baseVersion}"))
EOF
        docker build --network=host -t esp_server:${baseVersion} - < .ci/Dockerfile.1
        rm .ci/Dockerfile.1 .ci/Dockerfile.1.dcp
    fi

    state=`get_state`
fi

post_state "$context" "$description" "$state"
