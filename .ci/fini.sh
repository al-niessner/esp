#! /usr/bin/env bash

. .ci/util.sh

if current_state
then
    echo ''
else
    echo 'exiting with code 1'
    exit 1
fi
