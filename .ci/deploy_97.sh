#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the new AE (base, cit, devel, server, tools, worker) to all mentors"
context="continuous-integration/97/esp-deploy"

post_state "$context" "$description" "$state"

lv=$(layer_versions 1)
cit_version="${lv:17:16}"
base_version="${lv:0:16}"

if current_state
then
    set -x
    tempfn=$(mktemp -p /proj/sdp/data/stg)
    # save the docker images to a tarball
    docker save -o ${tempfn} esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_server:${base_version} esp_tools:${base_version} esp_worker:latest
    # install all of the new images
    ssh mentor0 docker load -i ${tempfn}
    ssh mentor1 docker load -i ${tempfn}
    ssh mentor2 docker load -i ${tempfn}
    ssh mentor3 docker load -i ${tempfn}
    ssh mentor4 docker load -i ${tempfn}
    ssh mentor5 docker load -i ${tempfn}
    ssh mentor0 docker container prune -f
    ssh mentor0 docker image prune -f
    ssh mentor1 docker container prune -f
    ssh mentor1 docker image prune -f
    ssh mentor2 docker container prune -f
    ssh mentor2 docker image prune -f
    ssh mentor3 docker container prune -f
    ssh mentor3 docker image prune -f
    ssh mentor4 docker container prune -f
    ssh mentor4 docker image prune -f
    ssh mentor5 docker container prune -f
    ssh mentor5 docker image prune -f
    # cleanup
    rm ${tempfn}
    set +x
    state=`get_state`
fi

post_state "$context" "$description" "$state"
