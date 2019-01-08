#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the new AE (base, cit, devel, server, tools, worker)"
context="continuous-integration/99/esp-deploy"

post_state "$context" "$description" "$state"

lv=$(layer_versions)
cit_version="${lv:17:16}"
py_version="${lv:0:16}"
version="$(lookup_version dawgie)"
[ -z "$(docker images | grep "esp_cit *${cit_version}")" ] && ( echo -n 'failure' > .ci/status.txt ; msg='CIT not found' ) || ( msg='CIT found' )
[ -z "$(docker images | grep "esp_devel *${version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg='${msg}\nDEV not found' ) || ( msg='${msg}\nDEV found')
[ -z "$(docker images | grep "esp_py *${py_version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg='${msg}\nPY not found' ) || ( msg='${msg}\nPY found')
[ -z "$(docker images | grep "esp_server *${version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg='${msg}\nSERVER not found' ) || ( msg='${msg}\nSERVER found')
[ -z "$(docker images | grep "esp_tools *${version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg='${msg}\nTOOLS not found' ) || ( msg='${msg}\nTOOLS found')
[ -z "$(docker images | grep "esp_worker *latest")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg='${msg}\nWORKER not found' ) || ( msg='${msg}\nWORKER found')

if current_state
then
    tempfn=$(mktemp -p /proj/sdp/data/stg)
    docker save -o ${tempfn} esp_cit:${cit_version} esp_devel:${version} esp_py:${py_version} esp_server:${version} esp_tools:${version} esp_worker:latest
    ssh mentor0 docker rmi esp_cit:${cit_version} esp_devel:${version} esp_py:${py_version} esp_tools:${version} esp_worker:latest
    ssh mentor1 docker rmi esp_cit:${cit_version} esp_devel:${version} esp_py:${py_version} esp_tools:${version} esp_worker:latest
    ssh mentor2 docker rmi esp_cit:${cit_version} esp_devel:${version} esp_py:${py_version} esp_tools:${version} esp_worker:latest
    ssh mentor3 docker rmi esp_cit:${cit_version} esp_devel:${version} esp_py:${py_version} esp_tools:${version} esp_worker:latest
    ssh mentor0 docker load -i ${tempfn}
    ssh mentor1 docker load -i ${tempfn}
    ssh mentor2 docker load -i ${tempfn}
    ssh mentor3 docker load -i ${tempfn}
    curl -XPOST http://mentor.jpl.nasa.gov:8080/app/submit?changeset=$(git rev-parse HEAD)\&submission=now > /dev/null 2>&1
    rm ${tempfn}
    state=`get_state`
else
    sendmail -f no-reply@esp.jpl.nasa.gov sdp@jpl.nasa.gov<<EOF
Subject: esp not-released

${msg}

EOF
fi

post_state "$context" "$description" "$state"
