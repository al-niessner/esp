#! /usr/bin/env bash

. .ci/util.sh

echo "BUILD_URL: ${BUILD_URL}"
echo "bn: ${BUILD_URL##*ESP-PR/}"
echo "ghprbPullId: ${ghprbPullId}"
bn=${BUILD_URL##*ESP-PR/}
curl ${BUILD_URL}/consoleText > /proj/sdp/logs/jenkins/pr.${ghprbPullId}.b.${bn%/*}.txt

if current_state
then
    echo ''
else
    echo 'exiting with code 1'
    exit 1
fi
