#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Wait for pipeline to process new AE"
context="continuous-integration/99/esp-deploy"

post_state "$context" "$description" "$state"

if current_state
then
    until python3 <<EOF
import json
import sys
try:
    data = json.loads ('$(curl --connect-timeout 10 --expect100-timeout 10 -XGET https://excalibur.jpl.nasa.gov:8080/app/pl/state)')
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
    .ci/update_runids.py -L /proj/sdp/data/logs -l ops.log -m /proj/sdp/ops/front-end/markdown/about.md
    sendmail -f no-reply@esp.jpl.nasa.gov sdp@jpl.nasa.gov<<EOF
Subject: esp released

At release $(git rev-parse HEAD)
Updated https://excalibur.jpl.nasa.gov:8080/pages/about.html

EOF

    state=`get_state`
fi

post_state "$context" "$description" "$state"
