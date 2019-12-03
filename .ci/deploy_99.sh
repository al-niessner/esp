#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Deploy the new AE (base, cit, devel, server, tools, worker)"
context="continuous-integration/99/esp-deploy"

post_state "$context" "$description" "$state"

lv=$(layer_versions 1)
cit_version="${lv:17:16}"
base_version="${lv:0:16}"
[ -z "$(docker images | grep "esp_cit *${cit_version}")" ] && ( echo -n 'failure' > .ci/status.txt ; msg="CIT not found" ) || msg="CIT found"
[ -z "$(docker images | grep "esp_devel *${base_version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg="${msg}\nDEV not found" ) || msg="${msg}\nDEV found"
[ -z "$(docker images | grep "esp_base *${base_version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg="${msg}\nPY not found" ) || msg="${msg}\nPY found"
[ -z "$(docker images | grep "esp_server *${base_version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg="${msg}\nSERVER not found" ) || msg="${msg}\nSERVER found"
[ -z "$(docker images | grep "esp_tools *${base_version}")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg="${msg}\nTOOLS not found" ) || msg="${msg}\nTOOLS found"
[ -z "$(docker images | grep "esp_worker *latest")" ] && ( echo -n 'failure' > .ci/status.txt  ; msg="${msg}\nWORKER not found" ) || msg="${msg}\nWORKER found"
echo "msg: ${msg}"

if current_state
then
    tempfn=$(mktemp -p /proj/sdp/data/stg)
    # save the docker images to a tarball
    docker save -o ${tempfn} esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_server:${base_version} esp_tools:${base_version} esp_worker:latest
    # delete unused copies of same tags
    ssh mentor0 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest
    ssh mentor1 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest
    ssh mentor2 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest
    ssh mentor3 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest
    ssh mentor4 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest
    ssh mentor5 docker rmi --force esp_cit:${cit_version} esp_devel:${base_version} esp_base:${base_version} esp_tools:${base_version} esp_worker:latest

    # install all of the new images
    ssh mentor0 docker load -i ${tempfn}
    ssh mentor1 docker load -i ${tempfn}
    ssh mentor2 docker load -i ${tempfn}
    ssh mentor3 docker load -i ${tempfn}
    ssh mentor4 docker load -i ${tempfn}
    ssh mentor5 docker load -i ${tempfn}
    # notify the pipeline that it is ready
    curl -XPOST http://mentor.jpl.nasa.gov:8080/app/submit?changeset=$(git rev-parse HEAD)\&submission=now > /dev/null 2>&1
    # cleanup
    rm ${tempfn}
    sleep 10

    if [[ $USER == dawgie-bot ]]
    then
        ssh mentor0 ${HOME}/docker_scrub.sh
        ssh mentor1 ${HOME}/docker_scrub.sh
        ssh mentor2 ${HOME}/docker_scrub.sh
        ssh mentor3 ${HOME}/docker_scrub.sh
        ssh mentor4 ${HOME}/docker_scrub.sh
        ssh mentor5 ${HOME}/docker_scrub.sh
    else
        ssh mentor0 ${PWD}/.ci/docker_scrub.sh
        ssh mentor1 ${PWD}/.ci/docker_scrub.sh
        ssh mentor2 ${PWD}/.ci/docker_scrub.sh
        ssh mentor3 ${PWD}/.ci/docker_scrub.sh
        ssh mentor4 ${PWD}/.ci/docker_scrub.sh
        ssh mentor5 ${PWD}/.ci/docker_scrub.sh
    fi

    until python3 <<EOF
import json
import sys

try:
    data = json.loads ('$(curl --connect-timeout 10 --expect100-timeout 10 -XGET http://mentor0.jpl.nasa.gov:8080/app/pl/state)')

    if data['name'] == 'running' and data['status'] == 'active':
        print ('match')
        sys.exit(0)
    else:
        print ('no match')
        sys.exit (1)
except Exception as ex:
    print ('exception', ex)
    sys.exit(2)
EOF
    do
        sleep 600
    done
    ${PWD}/.ci/update_runids.py -L /proj/sdp/data/logs -l ops.log -m /proj/sdp/ops/front-end/markdown/about.md

    state=`get_state`
else
    sendmail -f no-reply@esp.jpl.nasa.gov sdp@jpl.nasa.gov<<EOF
Subject: esp not-released

${msg}

EOF
fi

post_state "$context" "$description" "$state"
