#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Build a tools to manage the AE"
context="continuous-integration/01/esp-build-tools"

post_state "$context" "$description" "$state"

if current_state
then
    lv=$(layer_versions 1)
    baseVersion="${lv:0:16}"
            
    if [ -z "$(docker images | grep esp_tools:${version})" ]
    then
        python3 <<EOF
with open ('.ci/Dockerfile.tools', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.1', 'tw') as f: f.write (text.replace ("ghrVersion", "${baseVersion}"))
EOF
        docker build --network=host -t esp_tools:${baseVersion} - < .ci/Dockerfile.1
        rm .ci/Dockerfile.1 
    fi
    state=`get_state`
fi

post_state "$context" "$description" "$state"
