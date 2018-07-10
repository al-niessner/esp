#! /usr/bin/env bash

. .ci/util.sh

bn=${BUILD_URL##*ESP-PR/}
curl ${BUILD_URL}/consoleText > /proj/sdp/logs/jenkins/pr.${ghprbPullId}.b.${bn%/*}.txt

if current_state
then
    echo ''
else
    echo 'exiting with code 1'
    exit 1
fi
