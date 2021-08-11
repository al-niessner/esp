#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Verify the new AE (base, cit, devel, server, tools, worker) were built"
context="continuous-integration/96/esp-deploy"

post_state "$context" "$description" "$state"

lv=$(layer_versions 1)
cit_version="${lv:17:16}"
base_version="${lv:0:16}"

if current_state
then

    if [ -z "$(docker images | grep "esp_cit *${cit_version}")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="CIT not found"
    else
        msg="CIT found"
    fi
    
    if [ -z "$(docker images | grep "esp_devel *${base_version}")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="${msg}\nDEV not found"
    else
        msg="${msg}\nDEV found"
    fi

    if [ -z "$(docker images | grep "esp_base *${base_version}")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="${msg}\nPY not found"
    else
        msg="${msg}\nPY found"
    fi
    
    if [ -z "$(docker images | grep "esp_server *${base_version}")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="${msg}\nSERVER not found"
    else
        msg="${msg}\nSERVER found"
    fi

    
    if [ -z "$(docker images | grep "esp_tools *${base_version}")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="${msg}\nTOOLS not found"
    else
        msg="${msg}\nTOOLS found"
    fi
    
    if [ -z "$(docker images | grep "esp_worker *latest")" ]
    then
        echo -n 'failure' > .ci/status.txt
        msg="${msg}\nWORKER not found"
    else
        msg="${msg}\nWORKER found"
    fi

    echo "msg: ${msg}"

    if [ -n "$(echo $msg | grep not)" ]
    then
        echo "sending email of failure"
        sendmail -f no-reply@esp.jpl.nasa.gov sdp@jpl.nasa.gov<<EOF
Subject: esp not-released

$(echo ${msg})

EOF
    fi
    state=`get_state`
fi

post_state "$context" "$description" "$state"
