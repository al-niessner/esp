#! /usr/bin/env bash

if [ -f /tmp/crew.ops ]
then
    flock --unlock /tmp/crew.lock $(dirname $0)/spawn.sh
fi
