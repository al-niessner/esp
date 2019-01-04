#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Build a tools to manage the AE"
context="continuous-integration/01/esp-deploy-tools"

post_state "$context" "$description" "$state"

if current_state
then
    for destination in `destinations`
    do
        if [ "$destination" == "dawgie" ]
        then
            lv=$(layer_versions 1)
            source="`lookup_source $destination`"
            version="`lookup_version $destination`"
            layer_version="${lv:0:16}"
            
            if [ -z "$(docker images | grep esp_tools:${version})" ]
            then
               python3 <<EOF
with open ('.ci/Dockerfile.tools', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.1', 'tw') as f: f.write (text.replace ("ghrVersion", "${layer_version}"))
EOF
               docker build --network=host -t esp_tools:${version} - < .ci/Dockerfile.1
               rm .ci/Dockerfile.1 
            fi
        fi
    done
    state=`get_state`
fi

post_state "$context" "$description" "$state"
