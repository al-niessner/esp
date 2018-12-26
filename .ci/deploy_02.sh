#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Build a worker to execute the AE"
context="continuous-integration/01/esp-build-worker"

post_state "$context" "$description" "$state"

if current_state
then
    if [ -n "$(docker images | grep esp_worker | grep latest)" ]
    then
        docker rmi esp_worker:latest
    fi

    lv=$(layer_versions)
    version="${lv:0:16}"
    esp_git_rev="$(git rev-parse HEAD)"
    rm .ci/Dockerfile.1 .ci/Dockerfile.2
    python3 <<EOF
with open ('.ci/Dockerfile.worker', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.1', 'tw') as f: f.write (text.replace ("ghrVersion", "${version}").replace ('esp-git-rev', "${esp_git_rev}"))
with open ('setup.py', 'rt') as f: text = f.read()
with open ('setup.py', 'tw') as f: f.write (text.replace ('esp-git-rev', "${esp_git_rev}"))
EOF
    .ci/dcp.py --server .ci/Dockerfile.1 &
    while [ ! -f .ci/Dockerfile.1.dcp ]
    do
        sleep 3
    done
    docker build --network=host -t esp_worker:latest - < .ci/Dockerfile.1.dcp
    git checkout setup.py
    rm .ci/Dockerfile.1 .ci/Dockerfile.1.dcp
    state=`get_state`
fi

post_state "$context" "$description" "$state"
